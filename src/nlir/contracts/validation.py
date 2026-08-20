"""Atomic validation for externally supplied, source-backed IR fragments."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import ValidationError, model_validator

from nlir.contracts.common import SourceSpan, StrictFrozenModel
from nlir.contracts.diagnostics import DiagnosticSeverity, ValidationDiagnostic
from nlir.contracts.ir import IRFragment

if TYPE_CHECKING:
    from nlir.artifacts.models import SourceArtifact


class ValidationResult(StrictFrozenModel):
    """The validation boundary's all-or-nothing result."""

    fragment: IRFragment | None
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def must_not_mix_rejected_and_accepted_semantics(self) -> ValidationResult:
        has_error = any(
            diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in self.diagnostics
        )
        if has_error and self.fragment is not None:
            raise ValueError("an accepted fragment cannot include error diagnostics")
        if self.fragment is None and not has_error:
            raise ValueError("a rejected fragment must include an error diagnostic")
        return self


def validate_fragment(
    raw_fragment: object,
    artifacts: Mapping[str, SourceArtifact],
) -> ValidationResult:
    """Accept an entire fragment only after strict parsing and intrinsic checks.

    This boundary intentionally performs no repair or normalization.  Callers get
    the parsed fragment exactly as Pydantic produced it, or an error-only result
    with no semantic facts.
    """
    try:
        fragment = _parse_fragment(raw_fragment)
    except (TypeError, ValueError, ValidationError) as error:
        locations = error.errors() if isinstance(error, ValidationError) else ()
        is_evidence_error = any("evidence" in location["loc"] for location in locations)
        return _rejected(
            "invalid_evidence_span" if is_evidence_error else "invalid_ir_shape",
            (
                "Evidence does not conform to the strict source-span contract."
                if is_evidence_error
                else "IR fragment does not conform to the strict versioned schema."
            ),
            _fallback_span(artifacts),
        )

    fallback_span = _fallback_span(artifacts)
    diagnostics = [
        *_evidence_diagnostics(fragment, artifacts, fallback_span),
        *_duplicate_id_diagnostics(fragment, artifacts, fallback_span),
        *_reference_diagnostics(fragment, artifacts, fallback_span),
    ]
    if diagnostics:
        return ValidationResult(fragment=None, diagnostics=tuple(diagnostics))
    return ValidationResult(fragment=fragment)


def _parse_fragment(raw_fragment: object) -> IRFragment:
    if isinstance(raw_fragment, IRFragment):
        return raw_fragment
    if isinstance(raw_fragment, (str, bytes, bytearray)):
        return IRFragment.model_validate_json(raw_fragment)
    return IRFragment.model_validate_json(json.dumps(raw_fragment))


def _evidence_diagnostics(
    fragment: IRFragment,
    artifacts: Mapping[str, SourceArtifact],
    fallback_span: SourceSpan | None,
) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    for record in (*fragment.entities, *fragment.operations, *fragment.relationships):
        for evidence in record.evidence:
            artifact = artifacts.get(evidence.artifact_id)
            if artifact is None:
                diagnostics.append(
                    _diagnostic(
                        "invalid_evidence_span",
                        "Evidence references an artifact that is not in the source registry.",
                        fallback_span,
                    )
                )
            elif evidence.end > len(artifact.text):
                diagnostics.append(
                    _diagnostic(
                        "invalid_evidence_span",
                        "Evidence span exceeds the exact source text boundary.",
                        _known_span_or_fallback(evidence, artifact, fallback_span),
                    )
                )
    return tuple(diagnostics)


def _duplicate_id_diagnostics(
    fragment: IRFragment,
    artifacts: Mapping[str, SourceArtifact],
    fallback_span: SourceSpan | None,
) -> tuple[ValidationDiagnostic, ...]:
    diagnostics: list[ValidationDiagnostic] = []
    seen_ids: set[str] = set()
    for record in (*fragment.entities, *fragment.operations):
        if record.id in seen_ids:
            diagnostics.append(
                _diagnostic(
                    "duplicate_semantic_id",
                    f"Semantic identifier {record.id!r} is declared more than once.",
                    _trusted_span_or_fallback(record.evidence, artifacts, fallback_span),
                )
            )
        seen_ids.add(record.id)
    return tuple(diagnostics)


def _reference_diagnostics(
    fragment: IRFragment,
    artifacts: Mapping[str, SourceArtifact],
    fallback_span: SourceSpan | None,
) -> tuple[ValidationDiagnostic, ...]:
    entity_ids = {entity.id for entity in fragment.entities}
    diagnostics: list[ValidationDiagnostic] = []
    for operation in fragment.operations:
        references = (operation.actor, *operation.inputs, *operation.outputs, operation.destination)
        for reference in references:
            if reference is not None and reference not in entity_ids:
                diagnostics.append(
                    _diagnostic(
                        "dangling_semantic_reference",
                        f"Operation {operation.id!r} references undeclared entity {reference!r}.",
                        _trusted_span_or_fallback(operation.evidence, artifacts, fallback_span),
                    )
                )
    for relationship in fragment.relationships:
        for reference in (relationship.source, relationship.target):
            if reference not in entity_ids:
                diagnostics.append(
                    _diagnostic(
                        "dangling_semantic_reference",
                        f"Relationship references undeclared entity {reference!r}.",
                        _trusted_span_or_fallback(relationship.evidence, artifacts, fallback_span),
                    )
                )
    return tuple(diagnostics)


def _fallback_span(artifacts: Mapping[str, SourceArtifact]) -> SourceSpan | None:
    """Select a stable source anchor for errors that have no valid raw span yet."""
    non_empty_artifacts = [artifact for artifact in artifacts.values() if len(artifact.text)]
    if not non_empty_artifacts:
        return None
    artifact = min(non_empty_artifacts, key=lambda item: item.artifact_id)
    return SourceSpan(artifact_id=artifact.artifact_id, start=0, end=1)


def _known_span_or_fallback(
    span: SourceSpan,
    artifact: SourceArtifact,
    fallback_span: SourceSpan | None,
) -> SourceSpan | None:
    return span if span.end <= len(artifact.text) else fallback_span


def _trusted_span_or_fallback(
    evidence: tuple[SourceSpan, ...],
    artifacts: Mapping[str, SourceArtifact],
    fallback_span: SourceSpan | None,
) -> SourceSpan | None:
    for span in evidence:
        artifact = artifacts.get(span.artifact_id)
        if artifact is not None and span.end <= len(artifact.text):
            return span
    return fallback_span


def _rejected(code: str, message: str, span: SourceSpan | None) -> ValidationResult:
    return ValidationResult(fragment=None, diagnostics=(_diagnostic(code, message, span),))


def _diagnostic(code: str, message: str, span: SourceSpan | None) -> ValidationDiagnostic:
    return ValidationDiagnostic(
        code=code, severity=DiagnosticSeverity.ERROR, message=message, span=span
    )
