"""Public API checks for the library-first NLIR workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlir import NLIR, LiftedIR
from nlir.artifacts.models import SourceArtifact
from support import fixture_nlir

ROOT = Path(__file__).parents[1]

KEPT_IR: list[LiftedIR] = []
"""Caller-owned lift results for this module. The library stores nothing."""


@pytest.fixture(autouse=True)
def _clear_kept_ir() -> None:
    KEPT_IR.clear()


def _catalog(tmp_path: Path, source: SourceArtifact) -> Path:
    """Write one fixture catalog that lifts this source into a package install."""
    catalog = tmp_path / f"catalog-{source.artifact_id[:12]}.json"
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
                                                "end": len(source.text),
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
    return catalog


def _prompt(tmp_path: Path, name: str, text: str) -> tuple[Path, SourceArtifact]:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path, SourceArtifact.from_text(text, source_name=name)


def test_public_api_reads_rules_lifts_files_and_hunts_the_returned_ir(tmp_path: Path) -> None:
    source_path, source = _prompt(tmp_path, "prompt.md", "Install demo-package.")

    with fixture_nlir(_catalog(tmp_path, source)) as nlir:
        rule = nlir.read_rule(ROOT / "rules" / "package-install.yaml")
        rules = nlir.read_rule_dir(ROOT / "rules")
        lifted = nlir.lift_file(source_path)
        report = nlir.run_rule(rule, lifted)

    assert isinstance(lifted, LiftedIR)
    assert NLIR.LiftedIR is LiftedIR
    assert rule.id in {item.id for item in rules}
    assert lifted.artifact_ids == (source.artifact_id,)
    assert [result.status for result in report.results] == ["HIT"]
    assert not list(tmp_path.rglob("*.sqlite3"))
    assert not (tmp_path / ".nlir").exists()


def test_public_api_hunts_across_several_lift_results_kept_by_the_caller(tmp_path: Path) -> None:
    first_path, first_source = _prompt(tmp_path, "first.md", "Install demo-package.")
    second_path, second_source = _prompt(tmp_path, "second.md", "Install other-package please.")

    with fixture_nlir(_catalog(tmp_path, first_source)) as nlir:
        KEPT_IR.append(nlir.lift_file(first_path))
    with fixture_nlir(_catalog(tmp_path, second_source)) as nlir:
        KEPT_IR.append(nlir.lift_file(second_path))
        rule = nlir.read_rule(ROOT / "rules" / "package-install.yaml")
        report = nlir.run_rule(rule, KEPT_IR)

    assert [result.status for result in report.results] == ["HIT", "HIT"]
    assert {result.artifact_id for result in report.results} == {
        first_source.artifact_id,
        second_source.artifact_id,
    }


def test_lifted_ir_survives_caller_serialization_and_still_hunts(tmp_path: Path) -> None:
    source_path, source = _prompt(tmp_path, "prompt.md", "Install demo-package.")
    store = tmp_path / "caller-owned.json"

    with fixture_nlir(_catalog(tmp_path, source)) as nlir:
        rule = nlir.read_rule(ROOT / "rules" / "package-install.yaml")
        store.write_text(nlir.lift_file(source_path).model_dump_json(), encoding="utf-8")
        restored = LiftedIR.model_validate_json(store.read_text(encoding="utf-8"))
        report = nlir.run_rule(rule, restored)

    assert restored.artifact_ids == (source.artifact_id,)
    assert [result.status for result in report.results] == ["HIT"]


def test_public_api_requires_an_explicit_lifter_for_lifting(tmp_path: Path) -> None:
    source_path, _ = _prompt(tmp_path, "prompt.md", "Prompt text.")

    with NLIR() as nlir:
        with pytest.raises(RuntimeError, match="configured lifter"):
            nlir.lift_file(source_path)
