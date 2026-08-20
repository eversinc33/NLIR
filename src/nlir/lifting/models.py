"""Strict contracts shared by every semantic-lifter adapter."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import Field, model_validator

from nlir.canonical.models import (
    CanonicalFragment,
    NormalizationDiagnostic,
    SourceCanonicalId,
)
from nlir.contracts.common import SourceSpan, StrictFrozenModel
from nlir.contracts.diagnostics import DiagnosticSeverity
from nlir.contracts.ir import IRFragment

if TYPE_CHECKING:
    from nlir.artifacts.models import SourceArtifact
    from nlir.contracts.diagnostics import Diagnostic


class AttemptOutcome(StrEnum):
    """The externally observable state of one provider-shaped attempt."""

    FRAGMENT = "fragment"
    REFUSED = "refused"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class LifterStage(StrEnum):
    """The boundary stage responsible for an attempt diagnostic."""

    SETUP = "setup"
    SELECTION = "selection"
    LIFECYCLE = "lifecycle"
    VALIDATION = "validation"


class CanonicalAttemptStage(StrEnum):
    """The terminal audit stage after a lift attempt is considered canonically."""

    ACCEPTED = "accepted"
    LIFECYCLE_REJECTED = "lifecycle_rejected"
    VALIDATION_REJECTED = "validation_rejected"
    CANONICALIZATION_REJECTED = "canonicalization_rejected"


class LifterDiagnostic(StrictFrozenModel):
    """A non-semantic, typed rejection from the semantic-lifter boundary."""

    stage: LifterStage
    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    severity: DiagnosticSeverity
    message: Annotated[str, Field(min_length=1, max_length=512)]
    span: SourceSpan | None = None


class LiftAttemptResult(StrictFrozenModel):
    """All-or-nothing result for one ordered semantic-lifter attempt."""

    ordinal: Annotated[int, Field(ge=0)]
    outcome: AttemptOutcome | None = None
    fragment: IRFragment | None = None
    diagnostics: tuple[LifterDiagnostic, ...] = ()

    @model_validator(mode="after")
    def accepted_and_rejected_attempts_must_not_mix(self) -> LiftAttemptResult:
        has_error = any(
            diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in self.diagnostics
        )
        if self.fragment is not None and has_error:
            raise ValueError("an accepted attempt cannot include error diagnostics")
        if self.fragment is None and not has_error:
            raise ValueError("a rejected attempt must include an error diagnostic")
        return self


class CanonicalAttemptResult(StrictFrozenModel):
    """One ordered lift attempt after all canonical eligibility checks."""

    ordinal: Annotated[int, Field(ge=0)]
    stage: CanonicalAttemptStage
    outcome: AttemptOutcome | None = None
    canonical_fragment: CanonicalFragment | None = None
    source_to_canonical: tuple[SourceCanonicalId, ...] = ()
    diagnostics: tuple[LifterDiagnostic | NormalizationDiagnostic, ...] = ()

    @model_validator(mode="after")
    def must_be_complete_or_rejected(self) -> CanonicalAttemptResult:
        has_error = any(
            diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in self.diagnostics
        )
        if self.canonical_fragment is not None and has_error:
            raise ValueError("an accepted canonical attempt cannot include error diagnostics")
        if self.canonical_fragment is None:
            if not has_error:
                raise ValueError("a rejected canonical attempt must include an error diagnostic")
            if self.source_to_canonical:
                raise ValueError("a rejected canonical attempt cannot include source mappings")
        return self


@runtime_checkable
class SemanticLifter(Protocol):
    """Provider-shaped boundary implemented by every semantic-lifter adapter."""

    def lift(
        self,
        artifact: SourceArtifact,
        artifacts: Mapping[str, SourceArtifact],
    ) -> tuple[LiftAttemptResult, ...]:
        """Return every independently audited lift attempt in stable order."""


@runtime_checkable
class SemanticUnpacker(Protocol):
    """Optional live stage that creates inert virtual source artifacts."""

    def unpack(
        self, artifact: SourceArtifact
    ) -> tuple[tuple[SourceArtifact, ...], tuple[Diagnostic, ...]]:
        """Return model-derived children and non-finding diagnostics."""
