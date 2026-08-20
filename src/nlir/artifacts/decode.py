"""Bounded, deterministic transformations from encoded annotations to text children.

This module intentionally has no filesystem, process, network, import, or
evaluation behavior.  It only transforms text already held in a SourceArtifact.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from urllib.parse import unquote_to_bytes

from nlir.artifacts.extract import extract_annotations
from nlir.artifacts.models import (
    Annotation,
    AnnotationKind,
    DecodeCodec,
    DecodeLimits,
    DecodeProvenance,
    DecodeStep,
    SourceArtifact,
)
from nlir.contracts.diagnostics import Diagnostic, DiagnosticSeverity

_DECODABLE_KINDS = {
    AnnotationKind.BASE64_CANDIDATE: DecodeCodec.BASE64,
    AnnotationKind.HEX_CANDIDATE: DecodeCodec.HEX,
    AnnotationKind.URL_ENCODED_CANDIDATE: DecodeCodec.URL,
}


@dataclass(frozen=True, slots=True)
class VirtualChild:
    """One accepted child and its own non-semantic annotations."""

    artifact: SourceArtifact
    annotations: tuple[Annotation, ...]


@dataclass(frozen=True, slots=True)
class DecodeResult:
    """All virtual descendants and source-linked rejection diagnostics for one root."""

    children: tuple[VirtualChild, ...]
    diagnostics: tuple[Diagnostic, ...]


@dataclass(slots=True)
class _Budget:
    decoded_bytes: int = 0
    child_count: int = 0


def decode_artifact(root: SourceArtifact, *, limits: DecodeLimits | None = None) -> DecodeResult:
    """Expand eligible annotations recursively without interpreting their decoded text."""
    active_limits = limits or DecodeLimits()
    children: list[VirtualChild] = []
    diagnostics: list[Diagnostic] = []
    budget = _Budget()

    def walk(parent: SourceArtifact, depth: int) -> None:
        annotations = extract_annotations(parent)
        for annotation, codec in _decodable_annotations(annotations):
            candidate = annotation.span.extract(parent.text)
            decoded = _decode_candidate(candidate, codec, annotation, active_limits)
            if isinstance(decoded, Diagnostic):
                diagnostics.append(decoded)
                continue
            if depth >= active_limits.max_depth:
                diagnostics.append(_limit(annotation, "decode depth limit reached"))
                continue
            if budget.child_count >= active_limits.max_children:
                diagnostics.append(_limit(annotation, "virtual child limit reached"))
                continue
            if budget.decoded_bytes + len(decoded) > active_limits.max_aggregate_bytes:
                diagnostics.append(_limit(annotation, "aggregate decoded byte limit reached"))
                continue
            try:
                text = decoded.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                diagnostics.append(_non_text(annotation, "decoded bytes are not strict UTF-8 text"))
                continue
            if not text:
                diagnostics.append(_invalid(annotation, "decoded text is empty"))
                continue

            provenance = _provenance(parent, annotation, codec, depth + 1)
            child = SourceArtifact.from_virtual_text(text, decode_provenance=provenance)
            child_annotations = extract_annotations(child)
            children.append(VirtualChild(artifact=child, annotations=child_annotations))
            budget.child_count += 1
            budget.decoded_bytes += len(decoded)
            walk(child, depth + 1)

    walk(root, 0)
    return DecodeResult(children=tuple(children), diagnostics=tuple(diagnostics))


def _decode_candidate(
    candidate: str, codec: DecodeCodec, annotation: Annotation, limits: DecodeLimits
) -> bytes | Diagnostic:
    """Validate bounded ASCII syntax before performing one standard-library decode."""
    try:
        encoded = candidate.encode("ascii", errors="strict")
    except UnicodeEncodeError:
        return _invalid(annotation, "encoded candidate is not ASCII")
    if len(encoded) < limits.candidate_minimum_chars:
        return _limit(annotation, "encoded candidate is shorter than the configured minimum")
    if len(encoded) > limits.max_candidate_bytes:
        return _limit(annotation, "encoded candidate exceeds the input byte limit")
    if codec is DecodeCodec.BASE64 and (len(encoded) // 4) * 3 > limits.max_child_bytes:
        return _limit(annotation, "base64 candidate can exceed the decoded byte limit")
    if codec is DecodeCodec.HEX and len(encoded) // 2 > limits.max_child_bytes:
        return _limit(annotation, "hex candidate can exceed the decoded byte limit")

    try:
        if codec is DecodeCodec.BASE64:
            decoded = base64.b64decode(encoded, validate=True)
        elif codec is DecodeCodec.HEX:
            if len(encoded) % 2:
                return _invalid(annotation, "hex candidate has odd length")
            decoded = bytes.fromhex(candidate)
        else:
            if not _valid_percent_encoding(candidate):
                return _invalid(annotation, "percent candidate has invalid escape syntax")
            decoded = unquote_to_bytes(candidate)
    except (ValueError, binascii.Error):
        return _invalid(annotation, "candidate is not valid encoded data")

    if len(decoded) > limits.max_child_bytes:
        return _limit(annotation, "decoded bytes exceed the child byte limit")
    return decoded


def _decodable_annotations(
    annotations: tuple[Annotation, ...],
) -> tuple[tuple[Annotation, DecodeCodec], ...]:
    """Choose one deterministic codec for ambiguous identical spans.

    A hexadecimal token is also syntactically valid Base64 often enough to
    create duplicate children.  Prefer the more constrained hexadecimal form;
    all other candidate spans retain extractor order.
    """
    choices: list[tuple[Annotation, DecodeCodec]] = []
    seen_spans: set[tuple[int, int]] = set()
    for annotation in annotations:
        codec = _DECODABLE_KINDS.get(annotation.kind)
        if codec is None:
            continue
        span_key = (annotation.span.start, annotation.span.end)
        if span_key in seen_spans:
            continue
        if codec is DecodeCodec.BASE64 and any(
            other.kind is AnnotationKind.HEX_CANDIDATE
            and other.span.start == annotation.span.start
            and other.span.end == annotation.span.end
            for other in annotations
        ):
            continue
        seen_spans.add(span_key)
        choices.append((annotation, codec))
    return tuple(choices)


def _valid_percent_encoding(candidate: str) -> bool:
    return len(candidate) % 3 == 0 and all(
        candidate[index] == "%"
        and candidate[index + 1 : index + 3].isalnum()
        and all(
            character in "0123456789abcdef"
            for character in candidate[index + 1 : index + 3].lower()
        )
        for index in range(0, len(candidate), 3)
    )


def _provenance(
    parent: SourceArtifact, annotation: Annotation, codec: DecodeCodec, depth: int
) -> DecodeProvenance:
    chain: tuple[DecodeStep, ...] = ()
    if parent.decode_provenance is not None:
        prior = parent.decode_provenance
        chain = (
            *prior.chain,
            DecodeStep(
                parent_artifact_id=prior.parent_artifact_id,
                parent_span=prior.parent_span,
                codec=prior.codec,
            ),
        )
    return DecodeProvenance(
        parent_artifact_id=parent.artifact_id,
        parent_span=annotation.span,
        codec=codec,
        depth=depth,
        chain=chain,
    )


def _diagnostic(annotation: Annotation, code: str, message: str) -> Diagnostic:
    return Diagnostic(
        code=code,
        severity=DiagnosticSeverity.WARNING,
        message=message,
        span=annotation.span,
    )


def _invalid(annotation: Annotation, message: str) -> Diagnostic:
    return _diagnostic(annotation, "decode_invalid", message)


def _limit(annotation: Annotation, message: str) -> Diagnostic:
    return _diagnostic(annotation, "decode_limit", message)


def _non_text(annotation: Annotation, message: str) -> Diagnostic:
    return _diagnostic(annotation, "decode_non_text", message)
