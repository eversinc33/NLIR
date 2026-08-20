"""Optional real lift-and-rule checks for safe synthetic fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nlir.artifacts.loader import load_file
from nlir.artifacts.models import DecodeCodec
from nlir.ir import LiftedIR, hunt_records, lift_loaded_artifact
from nlir.lifting.live import LiveResponsesLifter
from nlir.rules.loader import load_rule

ROOT = Path(__file__).parents[2]
PACKAGE_SOURCE = ROOT / "tests" / "fixtures" / "live" / "package-install.md"
PACKAGE_RULE = ROOT / "rules" / "package-install.yaml"
BASE64_SOURCE = ROOT / "tests" / "fixtures" / "live" / "base64-hidden-command.md"
BASE64_RULE = ROOT / "rules" / "base64-hidden-command.yaml"
BENIGN_SKILL_SOURCE = ROOT / "tests" / "fixtures" / "live" / "benign-react-skill.md"
TRANSFER_SOURCE = ROOT / "tests" / "fixtures" / "live" / "indirect-data-transfer.md"
TRANSFER_POLICY_SOURCE = ROOT / "tests" / "fixtures" / "live" / "indirect-data-transfer-policy.md"
TRANSFER_RULE = ROOT / "rules" / "instruction-hijack-data-transfer.yaml"
SKILL_INSTALLER_SOURCE = ROOT / "tests" / "fixtures" / "live" / "openclaw-prerequisite-installer.md"
SKILL_INSTALLER_POLICY_SOURCE = (
    ROOT / "tests" / "fixtures" / "live" / "openclaw-prerequisite-installer-policy.md"
)
SKILL_INSTALLER_RULE = ROOT / "rules" / "openclaw-remote-installer.yaml"
MEMORY_SOURCE = ROOT / "tests" / "fixtures" / "live" / "openclaw-memory-persistence.md"
MEMORY_POLICY_SOURCE = (
    ROOT / "tests" / "fixtures" / "live" / "openclaw-memory-persistence-policy.md"
)
MEMORY_RULE = ROOT / "rules" / "openclaw-memory-persistence.yaml"
LINK_PREVIEW_SOURCE = ROOT / "tests" / "fixtures" / "live" / "openclaw-link-preview-transfer.md"
LINK_PREVIEW_POLICY_SOURCE = (
    ROOT / "tests" / "fixtures" / "live" / "openclaw-link-preview-transfer-policy.md"
)
LINK_PREVIEW_RULE = ROOT / "rules" / "credential-external-transfer.yaml"
REVERSED_OCTETS_SOURCE = ROOT / "tests" / "fixtures" / "live" / "parseltongue_reversed_octets.md"
MODEL_UNPACK_RULE = ROOT / "rules" / "hidden-command.yaml"


def _e2e_config() -> Path | None:
    """Return one caller-selected config only after all real-call gates pass."""
    if os.environ.get("NLIR_LIVE_E2E") != "1":
        return None
    if not os.environ.get("NLIR_LIVE_API_KEY", "").strip():
        return None
    value = os.environ.get("NLIR_LIVE_E2E_CONFIG", "")
    return Path(value) if value else None


def _live_report(source_path: Path) -> LiftedIR:
    """Lift one safe fixture and return its IR to this test only."""
    config = _e2e_config()
    if config is None:
        pytest.skip("set NLIR_LIVE_E2E=1, NLIR_LIVE_API_KEY, and NLIR_LIVE_E2E_CONFIG")
    lifter = LiveResponsesLifter.from_toml_file(config)
    metadata = lifter.lift_metadata()
    assert metadata is not None
    return LiftedIR(
        records=lift_loaded_artifact(load_file(source_path), lifter=lifter, metadata=metadata)
    )


def _rule(path: Path):
    """Load one tracked rule or stop the test with its typed reason."""
    loaded = load_rule(path)
    assert loaded.rule is not None, [diagnostic.code for diagnostic in loaded.diagnostics]
    return loaded.rule


def _attempt_states(report: LiftedIR) -> list[tuple[str, str, tuple[str, ...]]]:
    """Return safe attempt states without exposing fixture text or provider output."""
    return [
        (
            record.source.artifact_id,
            attempt.stage.value,
            tuple(diagnostic.code for diagnostic in attempt.diagnostics),
        )
        for record in report.records
        for attempt in record.canonical_attempts
    ]


@pytest.mark.live_e2e
def test_live_package_fixture_hits_semantic_install_rule() -> None:
    report = _live_report(PACKAGE_SOURCE)

    assert all(
        attempt.canonical_fragment is not None
        for record in report.records
        for attempt in record.canonical_attempts
    ), _attempt_states(report)
    results = hunt_records(report.records, _rule(PACKAGE_RULE)).results

    assert [result.status for result in results] == ["HIT"], _attempt_states(report)


@pytest.mark.live_e2e
def test_live_base64_fixture_hits_only_the_decoded_command() -> None:
    report = _live_report(BASE64_SOURCE)

    assert all(
        attempt.canonical_fragment is not None
        for record in report.records
        for attempt in record.canonical_attempts
    ), _attempt_states(report)
    sources = {record.source.artifact_id: record.source for record in report.records}
    results = hunt_records(report.records, _rule(BASE64_RULE)).results
    hits = [result for result in results if result.status == "HIT"]

    assert len(report.records) == 2, _attempt_states(report)
    assert len(hits) == 1, _attempt_states(report)
    assert hits[0].hints
    assert sources[hits[0].artifact_id].decode_provenance is not None
    assert sources[hits[0].artifact_id].decode_provenance.codec is DecodeCodec.BASE64


@pytest.mark.live_e2e
def test_live_indirect_transfer_pair_separates_instruction_hijack() -> None:
    positive = _live_report(TRANSFER_SOURCE)
    negative = _live_report(TRANSFER_POLICY_SOURCE)

    assert all(
        attempt.canonical_fragment is not None
        for record in positive.records
        for attempt in record.canonical_attempts
    ), _attempt_states(positive)
    assert all(
        attempt.canonical_fragment is not None
        for record in negative.records
        for attempt in record.canonical_attempts
    ), _attempt_states(negative)
    positive_results = hunt_records(positive.records, _rule(TRANSFER_RULE)).results
    negative_results = hunt_records(negative.records, _rule(TRANSFER_RULE)).results

    assert [result.status for result in positive_results] == ["HIT"], _attempt_states(positive)
    assert [result.status for result in negative_results] == ["NO_HIT"], _attempt_states(negative)


@pytest.mark.live_e2e
@pytest.mark.parametrize(
    ("source_path", "policy_path", "rule_path"),
    [
        (SKILL_INSTALLER_SOURCE, SKILL_INSTALLER_POLICY_SOURCE, SKILL_INSTALLER_RULE),
        (MEMORY_SOURCE, MEMORY_POLICY_SOURCE, MEMORY_RULE),
        (LINK_PREVIEW_SOURCE, LINK_PREVIEW_POLICY_SOURCE, LINK_PREVIEW_RULE),
    ],
)
def test_live_openclaw_attack_pairs_separate_direct_instructions(
    source_path: Path, policy_path: Path, rule_path: Path
) -> None:
    positive = _live_report(source_path)
    negative = _live_report(policy_path)

    assert all(
        attempt.canonical_fragment is not None
        for report in (positive, negative)
        for record in report.records
        for attempt in record.canonical_attempts
    ), (_attempt_states(positive), _attempt_states(negative))
    positive_results = hunt_records(positive.records, _rule(rule_path)).results
    negative_results = hunt_records(negative.records, _rule(rule_path)).results

    assert any(result.status == "HIT" for result in positive_results), _attempt_states(positive)
    assert all(result.status == "NO_HIT" for result in negative_results), _attempt_states(negative)


@pytest.mark.live_e2e
def test_live_reversed_octet_parseltongue_creates_a_model_unpacked_hit() -> None:
    report = _live_report(REVERSED_OCTETS_SOURCE)

    assert all(
        attempt.canonical_fragment is not None
        for record in report.records
        for attempt in record.canonical_attempts
    ), _attempt_states(report)
    sources = {record.source.artifact_id: record.source for record in report.records}
    hits = [
        result
        for result in hunt_records(report.records, _rule(MODEL_UNPACK_RULE)).results
        if result.status == "HIT"
    ]

    assert hits, _attempt_states(report)
    assert any(
        sources[hit.artifact_id].decode_provenance is not None
        and sources[hit.artifact_id].decode_provenance.codec is DecodeCodec.MODEL_INFERRED
        for hit in hits
    )


@pytest.mark.live_e2e
def test_live_benign_skill_fixture_hits_no_shipped_rule() -> None:
    """A large, realistic, non-adversarial skill file must not hit any rule."""
    report = _live_report(BENIGN_SKILL_SOURCE)

    assert all(
        attempt.canonical_fragment is not None
        for record in report.records
        for attempt in record.canonical_attempts
    ), _attempt_states(report)
    rules = [_rule(path) for path in sorted((ROOT / "rules").glob("*.yaml"))]
    hits = [
        (rule.id, result.artifact_id)
        for rule in rules
        for result in hunt_records(report.records, rule).results
        if result.status == "HIT"
    ]

    assert hits == [], _attempt_states(report)
