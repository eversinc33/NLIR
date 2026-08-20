"""Deterministic, one-artifact canonical semantic contracts."""

from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalFragment,
    CanonicalOperation,
    CanonicalRelationship,
    NormalizationDiagnostic,
    NormalizationResult,
    SourceCanonicalId,
)
from nlir.canonical.normalize import normalize_fragment

__all__ = [
    "CanonicalEntity",
    "CanonicalFragment",
    "CanonicalOperation",
    "CanonicalRelationship",
    "NormalizationDiagnostic",
    "NormalizationResult",
    "SourceCanonicalId",
    "normalize_fragment",
]
