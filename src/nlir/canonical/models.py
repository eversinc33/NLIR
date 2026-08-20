"""Strict immutable output contracts for canonical semantic facts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from nlir.contracts.common import ArtifactId, SourceSpan, StrictFrozenModel
from nlir.contracts.diagnostics import DiagnosticSeverity
from nlir.contracts.ir import (
    Confidence,
    EntityType,
    Modality,
    Opcode,
    RelationType,
    SemanticId,
    Sensitivity,
    TrustLevel,
)


class CanonicalEntity(StrictFrozenModel):
    """An evidence-preserving entity with a stable canonical identifier."""

    id: SemanticId
    type: EntityType
    subtype: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    value: Annotated[str, Field(min_length=1, max_length=256, pattern=r"^\S+$")] | None = None
    sensitivity: Sensitivity
    trust: TrustLevel
    evidence: tuple[SourceSpan, ...] = Field(min_length=1)
    confidence: Confidence
    underspecified: bool


class CanonicalOperation(StrictFrozenModel):
    """A declared operation with all entity references remapped canonically."""

    id: SemanticId
    op: Opcode
    actor: SemanticId | None = None
    inputs: tuple[SemanticId, ...] = ()
    outputs: tuple[SemanticId, ...] = ()
    destination: SemanticId | None = None
    modality: Modality
    evidence: tuple[SourceSpan, ...] = Field(min_length=1)
    confidence: Confidence
    underspecified: bool


class CanonicalRelationship(StrictFrozenModel):
    """A declared relationship between canonical entity identifiers."""

    id: SemanticId
    source: SemanticId
    relation: RelationType
    target: SemanticId
    evidence: tuple[SourceSpan, ...] = Field(min_length=1)
    confidence: Confidence
    underspecified: bool


class CanonicalFragment(StrictFrozenModel):
    """All canonical facts from exactly one source artifact."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_id: ArtifactId
    entities: tuple[CanonicalEntity, ...] = ()
    operations: tuple[CanonicalOperation, ...] = ()
    relationships: tuple[CanonicalRelationship, ...] = ()


class SourceCanonicalId(StrictFrozenModel):
    """One source semantic identifier and its canonical counterpart."""

    source_id: SemanticId
    canonical_id: SemanticId


class NormalizationDiagnostic(StrictFrozenModel):
    """A non-semantic explanation for an atomic canonicalization rejection."""

    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    severity: DiagnosticSeverity
    message: Annotated[str, Field(min_length=1, max_length=512)]
    span: SourceSpan | None = None


class NormalizationResult(StrictFrozenModel):
    """Either one complete canonical artifact fragment or error-only diagnostics."""

    fragment: CanonicalFragment | None
    source_to_canonical: tuple[SourceCanonicalId, ...] = ()
    diagnostics: tuple[NormalizationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def must_be_complete_or_error_only(self) -> NormalizationResult:
        has_error = any(
            diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in self.diagnostics
        )
        if self.fragment is not None and has_error:
            raise ValueError("an accepted canonical fragment cannot include error diagnostics")
        if self.fragment is None:
            if not has_error:
                raise ValueError("a rejected normalization must include an error diagnostic")
            if self.source_to_canonical:
                raise ValueError("a rejected normalization cannot include source mappings")
        return self
