"""Contract tests for the closed, hand-written rule format."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nlir.rules.models import Rule, RuleDiagnostic


def valid_rule() -> dict[str, object]:
    """Return the smallest complete rule with every version-one condition type."""
    return {
        "version": "1.0",
        "id": "credential-external-flow",
        "description": "Credential data reaches an external destination.",
        "select": {
            "credential": {"entity": {"sensitivity": "CREDENTIAL"}},
            "external": {"entity": {"trust": "EXTERNAL"}},
            "send": {"operation": {"op": "SEND"}},
        },
        "where": [
            {"direct": {"from": "credential", "to": "external", "relation": "SENT_TO"}},
            {"path": {"from": "credential", "to": "external", "kind": "relationship"}},
            {"sequence": {"from": "send", "to": "send"}},
            {"modality": {"selector": "send", "imperative": True, "quoted": False}},
            {"distance": {"from": "credential", "to": "external", "max_depth": 2}},
        ],
    }


def test_rule_model_accepts_the_small_closed_version_one_format() -> None:
    rule = Rule.model_validate_json(json.dumps(valid_rule()))

    assert rule.version == "1.0"
    assert tuple(rule.select) == ("credential", "external", "send")
    assert len(rule.where) == 5
    assert "severity" not in Rule.model_fields


def test_rule_diagnostics_do_not_return_severity() -> None:
    assert "severity" not in RuleDiagnostic.model_fields


def test_rule_metadata_keeps_human_context_outside_evaluation() -> None:
    raw = valid_rule()
    raw["metadata"] = {
        "description": "A documented semantic rule.",
        "author": "NLIR research fixtures",
        "references": ["https://example.invalid/research"],
    }

    rule = Rule.model_validate_json(json.dumps(raw))

    assert rule.metadata is not None
    assert rule.metadata.author == "NLIR research fixtures"
    assert rule.metadata.references == ("https://example.invalid/research",)
    assert "severity" not in type(rule.metadata).model_fields


def test_rule_model_accepts_any_selector_variants_of_one_record_kind() -> None:
    raw = valid_rule()
    raw["select"]["credential"] = {
        "any": [
            {"entity": {"type": "CREDENTIAL"}},
            {"entity": {"type": "SECRET"}},
        ]
    }

    rule = Rule.model_validate_json(json.dumps(raw))

    assert len(rule.select["credential"].any) == 2  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.update({"unknown": True}),
        lambda raw: raw.update({"version": "2.0"}),
        lambda raw: raw["select"]["credential"]["entity"].update({"trust": "NOT_A_TRUST"}),  # type: ignore[index]
        lambda raw: raw["select"].update({"bad-name": {"entity": {"trust": "EXTERNAL"}}}),  # type: ignore[index]
        lambda raw: raw["where"].append({"path": {"from": "credential", "to": "missing"}}),  # type: ignore[index]
        lambda raw: raw["where"].append({"expression": "credential and external"}),  # type: ignore[index]
        lambda raw: raw.update({"metadata": {"author": "missing description"}}),
        lambda raw: raw["select"].update(
            {
                "credential": {
                    "any": [
                        {"entity": {"type": "CREDENTIAL"}},
                        {"operation": {"op": "SEND"}},
                    ]
                }
            }
        ),  # type: ignore[index]
    ],
)
def test_rule_model_rejects_unknown_invalid_and_unsupported_forms(mutate: object) -> None:
    raw = valid_rule()
    mutate(raw)  # type: ignore[operator]

    with pytest.raises(ValidationError):
        Rule.model_validate_json(json.dumps(raw))
