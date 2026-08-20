"""Source-linked diagnostics that never become semantic IR facts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field

from nlir.contracts.common import SourceSpan, StrictFrozenModel


class DiagnosticSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class Diagnostic(StrictFrozenModel):
    """A bounded scanner/validation observation tied to source evidence."""

    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    severity: DiagnosticSeverity
    message: Annotated[str, Field(min_length=1, max_length=512)]
    span: SourceSpan


class ValidationDiagnostic(StrictFrozenModel):
    """A validation rejection that may lack any truthful source anchor.

    Scanner and decoder diagnostics always describe source observations, so they
    retain the required ``SourceSpan`` contract above.  Validation happens at an
    external boundary, however: malformed values and unresolved provenance can
    be rejected before a source span is known to be real.
    """

    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    severity: DiagnosticSeverity
    message: Annotated[str, Field(min_length=1, max_length=512)]
    span: SourceSpan | None = None
