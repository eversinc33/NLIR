"""End-to-end checks for the first three semantic hunting rules."""

from __future__ import annotations

import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest

from nlir.benchmark import LoadedBenchmarkCase, load_benchmark
from nlir.lifting.canonical import canonicalize_attempts
from nlir.lifting.models import CanonicalAttemptStage
from nlir.rules.evaluate import evaluate_rule
from nlir.rules.loader import load_rule
from nlir.rules.models import Rule, RuleResult
from support import FixtureLifter

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmark" / "manifest.json"
FIXTURE_CATALOG = ROOT / "tests" / "fixtures" / "lifting" / "fixture-attempts.json"
RULES = {
    "credential_external_transfer": (
        "credential-external-flow",
        ROOT / "tests" / "fixtures" / "rules" / "credential-external-flow.yaml",
    ),
    "decoded_instruction_action": (
        "decoded-instruction-execution",
        ROOT / "tests" / "fixtures" / "rules" / "decoded-instruction-execution.yaml",
    ),
    "transparency_suppression": (
        "transparency-suppression",
        ROOT / "tests" / "fixtures" / "rules" / "transparency-suppression.yaml",
    ),
}


def _no_side_effect(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("initial rule tests must stay offline and text-only")


@pytest.fixture(autouse=True)
def prevent_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject network and subprocess work during rule evaluation."""
    monkeypatch.setattr(socket, "create_connection", _no_side_effect)
    monkeypatch.setattr(subprocess, "run", _no_side_effect)
    monkeypatch.setattr(urllib.request, "urlopen", _no_side_effect)


def _canonical_case(
    case: LoadedBenchmarkCase,
    lifter: FixtureLifter,
):
    """Lift one benchmark text through the validated canonical boundary."""
    registry = {case.artifact.artifact_id: case.artifact}
    attempts = lifter.lift(case.artifact, registry)
    results = canonicalize_attempts(attempts, case.artifact, registry)
    accepted = [
        result.canonical_fragment
        for result in results
        if result.stage is CanonicalAttemptStage.ACCEPTED
    ]
    assert len(accepted) == 1, case.case.case_id
    assert accepted[0] is not None
    return accepted[0]


@pytest.mark.parametrize("family", sorted(RULES))
def test_initial_rules_hit_risky_cases_and_reject_paired_near_misses(family: str) -> None:
    """Each readable rule has one risky hit and six modality-aware no-hits."""
    corpus = load_benchmark(MANIFEST)
    lifter = FixtureLifter.from_json_file(FIXTURE_CATALOG)
    rule_id, rule_path = RULES[family]
    loaded_rule = load_rule(rule_path)

    assert loaded_rule.rule is not None
    assert loaded_rule.rule.id == rule_id
    assert "severity" not in loaded_rule.rule.model_dump()
    assert "severity" not in Rule.model_fields
    assert "severity" not in RuleResult.model_fields

    cases = [loaded for loaded in corpus.cases if loaded.case.family == family]
    risky = next(loaded for loaded in cases if loaded.case.role == "risky")
    assert rule_id in {assertion.rule_id for assertion in risky.case.expected_rules}

    first = evaluate_rule(loaded_rule.rule, _canonical_case(risky, lifter))
    second = evaluate_rule(loaded_rule.rule, _canonical_case(risky, lifter))
    assert first.model_dump_json() == second.model_dump_json()
    assert first.status == "HIT"
    assert first.matched_entity_ids or first.matched_operation_ids
    assert first.explanation is not None
    required_anchors = next(
        assertion.anchors for assertion in risky.case.expected_rules if assertion.rule_id == rule_id
    )
    assert all(
        any(
            span == risky.anchor_spans[anchor] for record in first.evidence for span in record.spans
        )
        for anchor in required_anchors
    )

    near_misses = [loaded for loaded in cases if loaded.case.role == "near_miss"]
    assert {loaded.case.modality for loaded in near_misses} == {
        "negation",
        "descriptive",
        "policy_detection",
        "quoted_example",
        "hypothetical",
        "attacker_description",
    }
    for near_miss in near_misses:
        assert near_miss.case.paired_with == risky.case.case_id
        assert rule_id in {assertion.rule_id for assertion in near_miss.case.forbidden_rules}
        result = evaluate_rule(loaded_rule.rule, _canonical_case(near_miss, lifter))
        assert result.status == "NO_HIT", near_miss.case.case_id
        assert result.matched_entity_ids == ()
        assert result.matched_operation_ids == ()
        assert result.matched_relationship_ids == ()
        assert result.evidence == ()
        assert result.explanation is None
