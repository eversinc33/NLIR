"""Immutable source artifacts and deterministic non-semantic observations."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from nlir.contracts.common import ArtifactId, SourceSpan, StrictFrozenModel
from nlir.contracts.diagnostics import Diagnostic


class ArtifactKind(StrEnum):
    PHYSICAL = "physical"
    VIRTUAL = "virtual"


class DecodeCodec(StrEnum):
    BASE64 = "base64"
    HEX = "hex"
    URL = "url"
    MODEL_INFERRED = "model_inferred"


class DecodeStep(StrictFrozenModel):
    """One immutable prior decoding hop for virtual-artifact provenance."""

    parent_artifact_id: ArtifactId
    parent_span: SourceSpan
    codec: DecodeCodec

    @model_validator(mode="after")
    def parent_span_must_belong_to_parent(self) -> DecodeStep:
        if self.parent_span.artifact_id != self.parent_artifact_id:
            raise ValueError("parent_span must reference parent_artifact_id")
        return self


class DecodeProvenance(DecodeStep):
    """Complete, bounded decode lineage for an inert virtual child."""

    depth: Annotated[int, Field(ge=1, le=16)]
    chain: tuple[DecodeStep, ...]
    method: Annotated[str, Field(min_length=1, max_length=80)] | None = None
    model_id: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    prompt_id: Annotated[str, Field(min_length=1, max_length=128)] | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None

    @model_validator(mode="after")
    def chain_must_precede_immediate_parent(self) -> DecodeProvenance:
        if len(self.chain) != self.depth - 1:
            raise ValueError("decode chain length must equal depth minus one")
        inferred = self.codec is DecodeCodec.MODEL_INFERRED
        has_model_details = all(
            value is not None
            for value in (self.method, self.model_id, self.prompt_id, self.confidence)
        )
        if inferred != has_model_details:
            raise ValueError("model-inferred provenance requires complete model details")
        return self


class DecodeLimits(StrictFrozenModel):
    """Versioned resource limits for pure virtual-child expansion."""

    version: Literal["1.0"] = "1.0"
    max_candidate_bytes: Annotated[int, Field(ge=1, le=1024 * 1024)] = 16 * 1024
    max_child_bytes: Annotated[int, Field(ge=1, le=1024 * 1024)] = 64 * 1024
    max_aggregate_bytes: Annotated[int, Field(ge=1, le=4 * 1024 * 1024)] = 256 * 1024
    max_depth: Annotated[int, Field(ge=1, le=16)] = 2
    max_children: Annotated[int, Field(ge=1, le=256)] = 16
    candidate_minimum_chars: Annotated[int, Field(ge=1, le=16 * 1024)] = 8

    @model_validator(mode="after")
    def aggregate_limit_must_cover_one_child(self) -> DecodeLimits:
        if self.max_aggregate_bytes < self.max_child_bytes:
            raise ValueError("max_aggregate_bytes must be at least max_child_bytes")
        if self.candidate_minimum_chars > self.max_candidate_bytes:
            raise ValueError("candidate_minimum_chars cannot exceed max_candidate_bytes")
        return self


class SourceArtifact(StrictFrozenModel):
    """Exact strict-UTF-8 source text with a SHA-256 content identity."""

    artifact_id: ArtifactId
    source_name: Annotated[str, Field(min_length=1, max_length=1024)]
    text: str
    kind: ArtifactKind = ArtifactKind.PHYSICAL
    decode_provenance: DecodeProvenance | None = None

    @staticmethod
    def content_id(text: str) -> str:
        """Return the SHA-256 identity of exact strict-UTF-8 source text."""
        return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_name: str,
        kind: ArtifactKind = ArtifactKind.PHYSICAL,
        decode_provenance: DecodeProvenance | None = None,
    ) -> SourceArtifact:
        """Build a canonical artifact in memory without filesystem I/O."""
        return cls(
            artifact_id=cls.content_id(text),
            source_name=source_name,
            text=text,
            kind=kind,
            decode_provenance=decode_provenance,
        )

    @classmethod
    def from_virtual_text(cls, text: str, *, decode_provenance: DecodeProvenance) -> SourceArtifact:
        """Build one inert, provenance-unique virtual source artifact."""
        artifact_id = cls.virtual_id(text, decode_provenance)
        return cls(
            artifact_id=artifact_id,
            source_name=f"virtual://{artifact_id}",
            text=text,
            kind=ArtifactKind.VIRTUAL,
            decode_provenance=decode_provenance,
        )

    @classmethod
    def virtual_id(cls, text: str, decode_provenance: DecodeProvenance) -> str:
        """Derive virtual identity from content and exact parent evidence."""
        return _stable_id(
            {
                "text_id": cls.content_id(text),
                "decode_provenance": decode_provenance.model_dump(mode="json", exclude_none=True),
            }
        )

    @model_validator(mode="after")
    def validate_identity_and_provenance(self) -> SourceArtifact:
        try:
            expected_id = self.content_id(self.text)
        except UnicodeEncodeError as error:
            raise ValueError("source text must be strict UTF-8 encodable") from error
        if self.kind is ArtifactKind.PHYSICAL and self.artifact_id != expected_id:
            raise ValueError("artifact_id must be the SHA-256 of exact source text")
        if self.kind is ArtifactKind.PHYSICAL and self.decode_provenance is not None:
            raise ValueError("physical artifacts cannot have decode provenance")
        if self.kind is ArtifactKind.VIRTUAL:
            if self.decode_provenance is None:
                raise ValueError("virtual artifacts require decode provenance")
            if not self.source_name.startswith("virtual://"):
                raise ValueError("virtual artifact source_name must use virtual://")
            virtual_id = self.virtual_id(self.text, self.decode_provenance)
            if self.artifact_id not in {expected_id, virtual_id}:
                raise ValueError("virtual artifact_id must match content or decode provenance")
        return self


class AnnotationKind(StrEnum):
    URL = "url"
    DOMAIN = "domain"
    IP_ADDRESS = "ip_address"
    FILESYSTEM_PATH = "filesystem_path"
    ENVIRONMENT_VARIABLE = "environment_variable"
    CODE_FENCE = "code_fence"
    SHELL_COMMAND = "shell_command"
    BASE64_CANDIDATE = "base64_candidate"
    HEX_CANDIDATE = "hex_candidate"
    URL_ENCODED_CANDIDATE = "url_encoded_candidate"
    HIGH_ENTROPY_CANDIDATE = "high_entropy_candidate"
    EXPLICIT_FILE_REFERENCE = "explicit_file_reference"


class Annotation(StrictFrozenModel):
    """A bounded deterministic observation; it contains no raw source candidate."""

    kind: AnnotationKind
    span: SourceSpan
    normalized_values: tuple[Annotated[str, Field(min_length=1, max_length=256)], ...] = ()

    @property
    def stable_id(self) -> str:
        return _stable_id(self.model_dump(mode="json"))


class ScanOccurrence(StrictFrozenModel):
    """A scanner occurrence wraps an annotation without creating semantics."""

    annotation: Annotation


class ScanReport(StrictFrozenModel):
    """Deterministic observations and diagnostics for a single source artifact."""

    artifact_id: ArtifactId
    occurrences: tuple[ScanOccurrence, ...]
    diagnostics: tuple[Diagnostic, ...]
    extractor_version: Annotated[str, Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def all_observations_must_belong_to_report_artifact(self) -> ScanReport:
        spans = [occurrence.annotation.span for occurrence in self.occurrences]
        spans.extend(diagnostic.span for diagnostic in self.diagnostics)
        if any(span.artifact_id != self.artifact_id for span in spans):
            raise ValueError("all scan observations must reference report artifact_id")
        return self

    @property
    def stable_id(self) -> str:
        return _stable_id(self.model_dump(mode="json"))


def _stable_id(value: object) -> str:
    """Hash canonical JSON identity material deterministically across processes."""
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8", errors="strict")).hexdigest()
