"""Behavioral contract for the browser view's analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlir.artifacts.models import SourceArtifact
from nlir.ir import LiftMetadata
from nlir.web.inspector import Inspector, ViewerInputError
from support import FixtureLifter

ROOT = Path(__file__).parents[2]
RULES_DIR = ROOT / "rules"


def _metadata() -> LiftMetadata:
    return LiftMetadata(
        ir_format="1.0",
        canonical_schema_version="1.0",
        normalizer_id="nlir.canonical.normalize:1.0",
        extractor_id="nlir.artifacts.extract:1.0",
        lifter_id="nlir.fixture_lifter:1.0",
        model_id="none",
        prompt_catalog_id="fixture-catalog-sha256:" + ("a" * 64),
    )


def _inspector(tmp_path: Path, prompt: str) -> Inspector:
    """Build an inspector whose lifter deterministically installs one package."""
    source = SourceArtifact.from_text(prompt, source_name="viewer-input")
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "version": "1.0",
                "fixtures": {
                    source.artifact_id: [
                        {
                            "outcome": "fragment",
                            "payload": {
                                "operations": [
                                    {
                                        "id": "install",
                                        "op": "INSTALL_PACKAGE",
                                        "actor": None,
                                        "inputs": [],
                                        "outputs": [],
                                        "destination": None,
                                        "modality": {
                                            "polarity": "positive",
                                            "imperative": True,
                                            "hypothetical": False,
                                            "conditional": False,
                                            "quoted": False,
                                            "example": False,
                                            "descriptive": False,
                                        },
                                        "evidence": [
                                            {
                                                "artifact_id": source.artifact_id,
                                                "start": 0,
                                                "end": len(prompt),
                                            }
                                        ],
                                        "confidence": 1.0,
                                        "underspecified": False,
                                    }
                                ]
                            },
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return Inspector(
        lifter=FixtureLifter.from_json_file(catalog),
        metadata=_metadata(),
        rules_directory=RULES_DIR,
    )


def test_list_rules_reads_every_local_rule_without_its_source_text(tmp_path: Path) -> None:
    inspector = _inspector(tmp_path, "Install demo-package.")

    rules = inspector.list_rules()

    assert {rule["id"] for rule in rules} == {path.stem for path in RULES_DIR.glob("*.yaml")}
    assert all(rule["description"] for rule in rules)


def test_rule_detail_returns_the_raw_yaml_text(tmp_path: Path) -> None:
    inspector = _inspector(tmp_path, "Install demo-package.")

    detail = inspector.rule_detail("package-install")

    assert detail["id"] == "package-install"
    assert "INSTALL_PACKAGE" in detail["text"]


def test_rule_detail_rejects_an_unknown_rule_id(tmp_path: Path) -> None:
    inspector = _inspector(tmp_path, "Install demo-package.")

    with pytest.raises(ViewerInputError):
        inspector.rule_detail("does-not-exist")


def test_analyze_rejects_blank_or_oversized_prompts(tmp_path: Path) -> None:
    inspector = _inspector(tmp_path, "Install demo-package.")

    with pytest.raises(ViewerInputError):
        inspector.analyze("   ")
    with pytest.raises(ViewerInputError):
        inspector.analyze("x" * (128 * 1024 + 1))


def test_analyze_links_tokens_graph_nodes_and_rule_matches_by_the_same_ids(
    tmp_path: Path,
) -> None:
    prompt = "Install demo-package."
    inspector = _inspector(tmp_path, prompt)

    result = inspector.analyze(prompt)

    assert len(result["tokens"]) == 1
    token = result["tokens"][0]
    assert (token["kind"], token["type"], token["start"], token["end"]) == (
        "operation",
        "INSTALL_PACKAGE",
        0,
        len(prompt),
    )
    token_node_id = token["node_id"]

    node_ids = {node["data"]["id"] for node in result["graph"]["nodes"]}
    assert token_node_id in node_ids
    assert result["graph"]["edges"] == []

    hit = next(rule for rule in result["rules"] if rule["id"] == "package-install")
    assert hit["status"] == "HIT"
    assert hit["matches"][0]["matched_node_ids"] == [token_node_id]
    assert hit["matches"][0]["spans"] == [{"start": 0, "end": len(prompt)}]

    no_hit_ids = {rule["id"] for rule in result["rules"] if rule["status"] == "NO_HIT"}
    assert no_hit_ids == {rule["id"] for rule in inspector.list_rules()} - {"package-install"}
