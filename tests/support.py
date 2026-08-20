"""Deterministic offline replay of the semantic-lifter boundary, for tests only.

The ``nlir`` library and CLI lift exclusively through a live model
(:class:`~nlir.lifting.live.LiveResponsesLifter`). This module exists so the
test suite can exercise lifting, canonicalization, and rule matching against
literal, checked-in IR without a live model call.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError, model_validator

from nlir import NLIR
from nlir.artifacts.models import SourceArtifact
from nlir.contracts.common import ArtifactId, StrictFrozenModel
from nlir.contracts.diagnostics import DiagnosticSeverity
from nlir.contracts.validation import validate_fragment
from nlir.ir import LiftMetadata
from nlir.lifting.models import (
    AttemptOutcome,
    LiftAttemptResult,
    LifterDiagnostic,
    LifterStage,
)


class FixtureAttempt(StrictFrozenModel):
    """One literal provider response retained in an offline JSON catalog."""

    outcome: AttemptOutcome
    payload: object | None = None

    @model_validator(mode="after")
    def payload_must_match_attempt_outcome(self) -> FixtureAttempt:
        if self.outcome is AttemptOutcome.FRAGMENT and self.payload is None:
            raise ValueError("fragment fixture attempts require a literal payload")
        if self.outcome is not AttemptOutcome.FRAGMENT and self.payload is not None:
            raise ValueError("only fragment fixture attempts may include a payload")
        return self


class FixtureCatalog(StrictFrozenModel):
    """Closed JSON fixture control-plane keyed by exact source artifact identity."""

    version: Literal["1.0"]
    fixtures: dict[ArtifactId, tuple[FixtureAttempt, ...]]

    @model_validator(mode="after")
    def each_registered_artifact_must_have_attempts(self) -> FixtureCatalog:
        if any(not attempts for attempts in self.fixtures.values()):
            raise ValueError("every registered artifact must declare at least one attempt")
        return self


@dataclass(frozen=True)
class FixtureLifter:
    """Load JSON attempts once and process exact artifact matches without side effects."""

    catalog: FixtureCatalog | None
    setup_diagnostic: LifterDiagnostic | None = None

    @classmethod
    def from_json_file(cls, path: str | Path) -> FixtureLifter:
        """Create a lifter whose catalog load failures are reported at ``lift()`` time."""
        try:
            raw_catalog = _load_json_object(Path(path))
            catalog = FixtureCatalog.model_validate_json(json.dumps(raw_catalog))
        except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as error:
            return cls(
                catalog=None, setup_diagnostic=_setup_diagnostic("invalid_fixture_catalog", error)
            )
        if not catalog.fixtures:
            return cls(
                catalog=None,
                setup_diagnostic=LifterDiagnostic(
                    stage=LifterStage.SETUP,
                    code="fixture_catalog_empty",
                    severity=DiagnosticSeverity.ERROR,
                    message="Fixture catalog does not register any artifact attempts.",
                ),
            )
        return cls(catalog=catalog)

    def lift(
        self,
        artifact: SourceArtifact,
        artifacts: Mapping[str, SourceArtifact],
    ) -> tuple[LiftAttemptResult, ...]:
        """Return every exact-match fixture attempt in its declared catalog order."""
        if self.setup_diagnostic is not None:
            return (_rejected(0, None, self.setup_diagnostic),)
        assert self.catalog is not None
        fixture_attempts = self.catalog.fixtures.get(artifact.artifact_id)
        if fixture_attempts is None:
            return (
                _rejected(
                    0,
                    None,
                    LifterDiagnostic(
                        stage=LifterStage.SELECTION,
                        code="fixture_not_found",
                        severity=DiagnosticSeverity.ERROR,
                        message="No fixture attempts are registered for this exact artifact ID.",
                    ),
                ),
            )
        return tuple(
            self._lift_attempt(ordinal, attempt.outcome, attempt.payload, artifacts)
            for ordinal, attempt in enumerate(fixture_attempts)
        )

    @staticmethod
    def _lift_attempt(
        ordinal: int,
        outcome: AttemptOutcome,
        payload: object | None,
        artifacts: Mapping[str, SourceArtifact],
    ) -> LiftAttemptResult:
        if outcome is AttemptOutcome.FRAGMENT:
            result = validate_fragment(payload, artifacts)
            if result.fragment is not None:
                return LiftAttemptResult(ordinal=ordinal, outcome=outcome, fragment=result.fragment)
            return LiftAttemptResult(
                ordinal=ordinal,
                outcome=outcome,
                diagnostics=tuple(
                    LifterDiagnostic(
                        stage=LifterStage.VALIDATION,
                        code=diagnostic.code,
                        severity=diagnostic.severity,
                        message=diagnostic.message,
                        span=diagnostic.span,
                    )
                    for diagnostic in result.diagnostics
                ),
            )
        return _rejected(
            ordinal,
            outcome,
            LifterDiagnostic(
                stage=LifterStage.LIFECYCLE,
                code=f"fixture_{outcome.value}",
                severity=DiagnosticSeverity.ERROR,
                message=f"Fixture attempt explicitly reported {outcome.value} output.",
            ),
        )


def fixture_nlir(catalog: str | Path) -> NLIR:
    """Build a library session backed by one offline fixture catalog, for tests only."""
    catalog_path = Path(catalog)
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    return NLIR(
        lifter=FixtureLifter.from_json_file(catalog_path),
        metadata=LiftMetadata(
            ir_format="1.0",
            canonical_schema_version="1.0",
            normalizer_id="nlir.canonical.normalize:1.0",
            extractor_id="nlir.artifacts.extract:1.0",
            lifter_id="nlir.fixture_lifter:1.0",
            model_id="none",
            prompt_catalog_id=f"fixture-catalog-sha256:{digest}",
        ),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one JSON control plane while refusing duplicate keys at every nesting level."""
    raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicates)
    if not isinstance(raw, dict):
        raise ValueError("fixture catalog must be a JSON object")
    return raw


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r} is not permitted")
        result[key] = value
    return result


def _setup_diagnostic(code: str, error: Exception) -> LifterDiagnostic:
    message = str(error).splitlines()[0] or "Fixture catalog could not be loaded."
    return LifterDiagnostic(
        stage=LifterStage.SETUP,
        code=code,
        severity=DiagnosticSeverity.ERROR,
        message=message[:512],
    )


def _rejected(
    ordinal: int,
    outcome: AttemptOutcome | None,
    diagnostic: LifterDiagnostic,
) -> LiftAttemptResult:
    return LiftAttemptResult(ordinal=ordinal, outcome=outcome, diagnostics=(diagnostic,))
