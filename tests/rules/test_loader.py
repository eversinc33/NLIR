"""Regression tests for bounded YAML rule loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from nlir.rules.loader import MAX_RULE_BYTES, load_rule, load_rule_text

ROOT = Path(__file__).parents[2]

VALID_RULE = """\
version: "1.0"
id: credential-external-flow
description: Credential data reaches an external destination.
select:
  credential:
    entity:
      sensitivity: CREDENTIAL
  external:
    entity:
      trust: EXTERNAL
  send:
    operation:
      op: SEND
where:
  - direct:
      from: credential
      to: external
      relation: SENT_TO
  - modality:
      selector: send
      imperative: true
      quoted: false
"""


def test_loader_accepts_one_small_readable_rule() -> None:
    result = load_rule_text(VALID_RULE, source_name="credential-flow.yaml")

    assert result.rule is not None
    assert result.rule.id == "credential-external-flow"
    assert result.diagnostics == ()


def test_package_install_rule_loads_with_the_specific_install_opcode() -> None:
    result = load_rule(ROOT / "rules" / "package-install.yaml")

    assert result.rule is not None
    assert result.rule.id == "package-install"
    assert result.rule.select["install"].operation.op.value == "INSTALL_PACKAGE"  # type: ignore[union-attr]


def test_base64_hidden_command_rule_loads_with_decode_provenance() -> None:
    result = load_rule(ROOT / "rules" / "base64-hidden-command.yaml")

    assert result.rule is not None
    assert result.rule.id == "base64-hidden-command"
    assert result.rule.where[0].decoded_from.codec.value == "base64"  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "name",
    [
        "base64-hidden-command",
        "configuration-change-persistence",
        "credential-external-transfer",
        "instruction-hijack-data-transfer",
        "hidden-command",
        "log-deletion",
        "openclaw-memory-persistence",
        "openclaw-remote-installer",
        "package-install",
        "skill-silent-package-install",
        "transparency-suppression",
    ],
)
def test_public_rules_keep_human_metadata(name: str) -> None:
    result = load_rule(ROOT / "rules" / f"{name}.yaml")

    assert result.rule is not None
    assert result.rule.metadata is not None
    assert result.rule.metadata.description
    assert result.rule.metadata.author


def test_loader_rejects_deeply_nested_yaml_with_a_typed_diagnostic() -> None:
    text = "value: " + "[" * 100 + "value" + "]" * 100 + "\n"

    result = load_rule_text(text, source_name="deep.yaml")

    assert result.rule is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["yaml_nesting_limit"]


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("", "empty_rule"),
        ("- item", "invalid_rule_root"),
        ("id: duplicate\nid: duplicate\n", "duplicate_yaml_key"),
        (
            VALID_RULE + "---\nversion: '1.0'\nid: second\nselect: {}\nwhere: []\n",
            "multiple_yaml_documents",
        ),
        ("value: &shared value\n" + VALID_RULE, "yaml_alias_not_allowed"),
        ("!custom\n" + VALID_RULE, "yaml_tag_not_allowed"),
        ("version: '1.0'\nid: bad\nselect: {}\nwhere: []\nunknown: true\n", "invalid_rule_shape"),
        ("version: [\n", "malformed_yaml"),
        ("x" * (MAX_RULE_BYTES + 1), "oversized_rule"),
    ],
)
def test_loader_rejects_unsafe_or_unsupported_yaml(text: str, code: str) -> None:
    result = load_rule_text(text, source_name="rule.yaml")

    assert result.rule is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == [code]
    assert all(diagnostic.span is None for diagnostic in result.diagnostics)
    assert all("severity" not in type(diagnostic).model_fields for diagnostic in result.diagnostics)
