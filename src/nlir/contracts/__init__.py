"""Strict, versioned contracts for NLIR."""

from nlir.contracts.common import SourceSpan
from nlir.contracts.diagnostics import Diagnostic, DiagnosticSeverity
from nlir.contracts.ir import IRFragment
from nlir.contracts.validation import ValidationResult, validate_fragment

__all__ = [
    "Diagnostic",
    "DiagnosticSeverity",
    "IRFragment",
    "SourceSpan",
    "ValidationResult",
    "validate_fragment",
]
