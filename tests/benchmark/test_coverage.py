"""Completeness checks for the frozen paired synthetic corpus."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from nlir.benchmark import load_benchmark

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "benchmark" / "manifest.json"
FAMILIES = {
    "credential_external_transfer",
    "package_install",
    "instruction_override_privileged_action",
    "decoded_instruction_action",
    "configuration_change_persistence",
    "openclaw_memory_persistence",
    "transparency_suppression",
}
SINGLE_CASE_FAMILIES = {
    "log_deletion",
    "parseltongue_binary_spacing",
    "parseltongue_reversed_octets",
    "parseltongue_custom_bijection",
}
NEAR_MISS_KINDS = {
    "negation",
    "descriptive",
    "policy_detection",
    "quoted_example",
    "hypothetical",
    "attacker_description",
}


def test_real_corpus_has_one_seed_and_every_near_miss_kind_per_family() -> None:
    corpus = load_benchmark(MANIFEST)
    grouped = defaultdict(list)
    for loaded in corpus.cases:
        grouped[loaded.case.family].append(loaded.case)

    assert set(grouped) == FAMILIES | SINGLE_CASE_FAMILIES
    assert len(corpus.cases) >= 50
    for family, cases in grouped.items():
        risky = [case for case in cases if case.role == "risky"]
        near_misses = [case for case in cases if case.role == "near_miss"]
        assert len(risky) == 1, family
        if family in SINGLE_CASE_FAMILIES:
            assert near_misses == [], family
        else:
            assert {case.modality for case in near_misses} == NEAR_MISS_KINDS, family
        assert all(case.paired_with == risky[0].case_id for case in near_misses), family


def test_every_case_has_evidence_backed_future_oracles() -> None:
    corpus = load_benchmark(MANIFEST)
    for loaded in corpus.cases:
        case = loaded.case
        assert case.expected_facts and case.forbidden_facts, case.case_id
        if case.role == "risky":
            assert case.expected_rules, case.case_id
        assert set(loaded.anchor_spans) == {"source", "action", "sink", "modality"}, case.case_id


def test_required_locations_languages_and_source_preservation_are_present() -> None:
    corpus = load_benchmark(MANIFEST)
    locations = {loaded.case.location for loaded in corpus.cases}
    languages = {loaded.case.language for loaded in corpus.cases}
    text_by_case = {loaded.case.case_id: loaded.artifact.text for loaded in corpus.cases}

    assert locations == {
        "markdown_body",
        "yaml_metadata",
        "json_string",
        "markdown_reference",
        "encoded_virtual_child",
    }
    assert languages == {"en", "de", "fr", "es"}
    assert any("\r\n" in text for text in text_by_case.values())
    assert any(not text.isascii() for text in text_by_case.values())
    assert "unresolved-context.txt" in text_by_case["configuration-change-persistence-quote"]
