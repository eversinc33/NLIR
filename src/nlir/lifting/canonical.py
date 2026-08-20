"""Canonicalize ordered lift attempts from any semantic lifter."""

from __future__ import annotations

from collections.abc import Mapping

from nlir.artifacts.models import SourceArtifact
from nlir.canonical.normalize import normalize_fragment
from nlir.lifting.models import (
    CanonicalAttemptResult,
    CanonicalAttemptStage,
    LiftAttemptResult,
    LifterStage,
)


def canonicalize_attempts(
    attempts: tuple[LiftAttemptResult, ...],
    artifact: SourceArtifact,
    artifacts: Mapping[str, SourceArtifact],
) -> tuple[CanonicalAttemptResult, ...]:
    """Report the canonical terminal state for every already ordered lift attempt."""
    results: list[CanonicalAttemptResult] = []
    for attempt in attempts:
        if attempt.fragment is None:
            results.append(
                CanonicalAttemptResult(
                    ordinal=attempt.ordinal,
                    stage=(
                        CanonicalAttemptStage.VALIDATION_REJECTED
                        if _is_validation_rejection(attempt)
                        else CanonicalAttemptStage.LIFECYCLE_REJECTED
                    ),
                    outcome=attempt.outcome,
                    diagnostics=attempt.diagnostics,
                )
            )
            continue
        normalized = normalize_fragment(attempt.fragment, artifact.artifact_id, artifacts)
        if normalized.fragment is None:
            results.append(
                CanonicalAttemptResult(
                    ordinal=attempt.ordinal,
                    stage=CanonicalAttemptStage.CANONICALIZATION_REJECTED,
                    outcome=attempt.outcome,
                    diagnostics=normalized.diagnostics,
                )
            )
            continue
        results.append(
            CanonicalAttemptResult(
                ordinal=attempt.ordinal,
                stage=CanonicalAttemptStage.ACCEPTED,
                outcome=attempt.outcome,
                canonical_fragment=normalized.fragment,
                source_to_canonical=normalized.source_to_canonical,
            )
        )
    return tuple(results)


def _is_validation_rejection(attempt: LiftAttemptResult) -> bool:
    return bool(attempt.diagnostics) and all(
        diagnostic.stage is LifterStage.VALIDATION for diagnostic in attempt.diagnostics
    )
