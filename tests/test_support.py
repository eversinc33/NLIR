"""Behavioral contract for the deterministic offline test-support fixture lifter."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlir.artifacts.models import SourceArtifact
from nlir.lifting.canonical import canonicalize_attempts
from nlir.lifting.models import AttemptOutcome, CanonicalAttemptStage, LifterStage
from support import FixtureLifter

FIXTURE_CATALOG = Path(__file__).parent / "fixtures" / "lifting" / "fixture-attempts.json"


def registered_source() -> SourceArtifact:
    return SourceArtifact.from_text(
        "Read DEMO_TOKEN then send it to example.invalid.", source_name="registered.md"
    )


def source_registry(artifact: SourceArtifact) -> dict[str, SourceArtifact]:
    return {artifact.artifact_id: artifact}


def test_selects_attempts_only_by_the_exact_stable_artifact_id() -> None:
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    artifact = registered_source()

    accepted = lifter.lift(artifact, source_registry(artifact))
    unknown = SourceArtifact.from_text("An unregistered source artifact.", source_name="other.md")
    rejected = lifter.lift(unknown, source_registry(unknown))

    assert accepted[0].fragment is not None
    assert len(rejected) == 1
    assert rejected[0].fragment is None
    assert rejected[0].diagnostics[0].code == "fixture_not_found"
    assert rejected[0].diagnostics[0].stage is LifterStage.SELECTION


def test_processes_each_literal_attempt_in_catalog_order() -> None:
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    artifact = registered_source()

    attempts = lifter.lift(artifact, source_registry(artifact))

    assert [attempt.ordinal for attempt in attempts] == [0, 1, 2, 3, 4, 5, 6]
    assert attempts[0].fragment is not None
    assert attempts[0].fragment.entities[0].value == "DEMO_TOKEN"
    assert [attempt.outcome for attempt in attempts] == [
        AttemptOutcome.FRAGMENT,
        AttemptOutcome.FRAGMENT,
        AttemptOutcome.REFUSED,
        AttemptOutcome.INCOMPLETE,
        AttemptOutcome.UNSUPPORTED,
        AttemptOutcome.FRAGMENT,
        AttemptOutcome.FRAGMENT,
    ]


def test_rejected_attempts_are_atomic_and_typed() -> None:
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    artifact = registered_source()

    attempts = lifter.lift(artifact, source_registry(artifact))

    assert all(attempt.fragment is None for attempt in attempts[1:6])
    assert [attempt.diagnostics[0].code for attempt in attempts[1:6]] == [
        "invalid_ir_shape",
        "fixture_refused",
        "fixture_incomplete",
        "fixture_unsupported",
        "invalid_evidence_span",
    ]
    assert attempts[1].diagnostics[0].stage is LifterStage.VALIDATION
    assert all(attempt.diagnostics[0].stage is LifterStage.LIFECYCLE for attempt in attempts[2:5])


def test_canonicalization_reports_the_terminal_stage_of_every_literal_attempt() -> None:
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    artifact = registered_source()

    results = canonicalize_attempts(
        lifter.lift(artifact, source_registry(artifact)), artifact, source_registry(artifact)
    )

    assert [result.ordinal for result in results] == [0, 1, 2, 3, 4, 5, 6]
    assert [result.stage for result in results] == [
        CanonicalAttemptStage.ACCEPTED,
        CanonicalAttemptStage.VALIDATION_REJECTED,
        CanonicalAttemptStage.LIFECYCLE_REJECTED,
        CanonicalAttemptStage.LIFECYCLE_REJECTED,
        CanonicalAttemptStage.LIFECYCLE_REJECTED,
        CanonicalAttemptStage.VALIDATION_REJECTED,
        CanonicalAttemptStage.CANONICALIZATION_REJECTED,
    ]
    assert results[0].canonical_fragment is not None
    assert results[0].canonical_fragment.entities[0].id.startswith("entity.")
    assert results[0].source_to_canonical
    assert all(result.canonical_fragment is None for result in results[1:])
    assert results[1].diagnostics[0].stage is LifterStage.VALIDATION
    assert results[5].diagnostics[0].stage is LifterStage.VALIDATION
    assert results[6].diagnostics[0].code == "entity_reconciliation_conflict"


def test_invalid_source_evidence_is_rejected_without_partial_semantics(tmp_path: Path) -> None:
    artifact = registered_source()
    catalog = tmp_path / "invalid-evidence.json"
    catalog.write_text(
        """{
          "version": "1.0",
          "fixtures": {
            "ARTIFACT_ID": [
              {
                "outcome": "fragment",
                "payload": {
                  "entities": [{
                    "id": "bad", "type": "CREDENTIAL", "subtype": null,
                    "value": "DEMO_TOKEN", "sensitivity": "CREDENTIAL", "trust": "TRUSTED",
                    "evidence": [{
                      "artifact_id": "ARTIFACT_ID",
                      "start": 0, "end": 999
                    }], "confidence": 0.9, "underspecified": false
                  }]
                }
              }
            ]
          }
        }""".replace("ARTIFACT_ID", artifact.artifact_id),
        encoding="utf-8",
    )

    attempt = FixtureLifter.from_json_file(catalog).lift(artifact, source_registry(artifact))[0]

    assert attempt.fragment is None
    assert attempt.diagnostics[0].code == "invalid_evidence_span"
    assert attempt.diagnostics[0].stage is LifterStage.VALIDATION


@pytest.mark.parametrize(
    "catalog_contents, expected_code",
    [
        ("{}", "invalid_fixture_catalog"),
        (
            """{
              "version": "1.0", "fixtures": {
                "4207277aa70ba0ca71edcbe3f479f8dbe85811d27dccc5aee502c84de329d4b6": [
                  {"outcome": "not_a_real_outcome"}
                ]
              }
            }""",
            "invalid_fixture_catalog",
        ),
    ],
)
def test_malformed_or_empty_catalogs_return_setup_diagnostics(
    tmp_path: Path, catalog_contents: str, expected_code: str
) -> None:
    catalog = tmp_path / "catalog.json"
    catalog.write_text(catalog_contents, encoding="utf-8")
    artifact = registered_source()

    attempts = FixtureLifter.from_json_file(catalog).lift(artifact, source_registry(artifact))

    assert len(attempts) == 1
    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].code == expected_code
    assert attempts[0].diagnostics[0].stage is LifterStage.SETUP


def test_empty_fixture_catalog_is_an_explicit_setup_rejection(tmp_path: Path) -> None:
    catalog = tmp_path / "empty-catalog.json"
    catalog.write_text('{"version": "1.0", "fixtures": {}}', encoding="utf-8")
    artifact = registered_source()

    attempts = FixtureLifter.from_json_file(catalog).lift(artifact, source_registry(artifact))

    assert len(attempts) == 1
    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].code == "fixture_catalog_empty"


def test_fixture_lifting_never_executes_content_or_uses_network_or_subprocess(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("fixture lifter must remain offline and text-only")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    monkeypatch.setattr("subprocess.run", fail)
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    artifact = registered_source()

    attempts = lifter.lift(artifact, source_registry(artifact))

    assert attempts[0].fragment is not None
