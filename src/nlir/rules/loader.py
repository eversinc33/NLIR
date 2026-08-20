"""Bounded loading for one strict YAML rule document."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError
from yaml.nodes import MappingNode, Node
from yaml.tokens import (
    AliasToken,
    AnchorToken,
    FlowMappingEndToken,
    FlowMappingStartToken,
    FlowSequenceEndToken,
    FlowSequenceStartToken,
    TagToken,
)

from nlir.rules.models import Rule, RuleDiagnostic, RuleLoadResult

MAX_RULE_BYTES = 64 * 1024
"""Maximum UTF-8 byte size for one rule document."""

MAX_YAML_NESTING = 64
"""Maximum allowed YAML flow-collection nesting depth."""


def load_rule(path: Path, *, max_bytes: int = MAX_RULE_BYTES) -> RuleLoadResult:
    """Load one local rule file without reading more than its byte limit."""
    try:
        size = path.stat().st_size
    except OSError:
        return _rejected("unreadable_rule", "The rule file cannot be read.")
    if size > max_bytes:
        return _rejected("oversized_rule", "The rule file exceeds the configured byte limit.")
    try:
        payload = path.read_bytes()
    except OSError:
        return _rejected("unreadable_rule", "The rule file cannot be read.")
    return load_rule_text(payload, source_name=path.name, max_bytes=max_bytes)


def load_rule_text(
    text: str | bytes,
    *,
    source_name: str = "<rule>",
    max_bytes: int = MAX_RULE_BYTES,
) -> RuleLoadResult:
    """Load one mapping document from text under a strict byte limit.

    ``source_name`` is accepted for callers that track rule files. Diagnostics do
    not claim source positions because this contract does not retain YAML spans.
    """
    del source_name
    if max_bytes < 1:
        raise ValueError("max_bytes must be at least one")
    try:
        payload = text.encode("utf-8") if isinstance(text, str) else bytes(text)
        decoded = payload.decode("utf-8")
    except UnicodeDecodeError:
        return _rejected("invalid_rule_encoding", "The rule file is not valid UTF-8 text.")
    if len(payload) > max_bytes:
        return _rejected("oversized_rule", "The rule file exceeds the configured byte limit.")
    if not decoded.strip():
        return _rejected("empty_rule", "The rule file is empty.")
    try:
        tokens = tuple(yaml.scan(decoded, Loader=yaml.SafeLoader))
    except RecursionError:
        return _rejected("yaml_nesting_limit", "The rule file exceeds the YAML nesting limit.")
    except yaml.YAMLError:
        return _rejected("malformed_yaml", "The rule file is not valid YAML.")
    if _exceeds_flow_nesting(tokens):
        return _rejected("yaml_nesting_limit", "The rule file exceeds the YAML nesting limit.")
    if any(isinstance(token, (AliasToken, AnchorToken)) for token in tokens):
        return _rejected("yaml_alias_not_allowed", "YAML anchors and aliases are not allowed.")
    if any(isinstance(token, TagToken) for token in tokens):
        return _rejected("yaml_tag_not_allowed", "YAML tags are not allowed.")
    try:
        documents = tuple(yaml.compose_all(decoded, Loader=yaml.SafeLoader))
    except RecursionError:
        return _rejected("yaml_nesting_limit", "The rule file exceeds the YAML nesting limit.")
    except yaml.YAMLError:
        return _rejected("malformed_yaml", "The rule file is not valid YAML.")
    if len(documents) != 1:
        return _rejected(
            "multiple_yaml_documents", "A rule file must contain exactly one document."
        )
    root = documents[0]
    if root is None:
        return _rejected("empty_rule", "The rule file is empty.")
    if not isinstance(root, MappingNode):
        return _rejected("invalid_rule_root", "The rule document root must be a mapping.")
    try:
        has_duplicate_key = _has_duplicate_key(root)
    except RecursionError:
        return _rejected("yaml_nesting_limit", "The rule file exceeds the YAML nesting limit.")
    if has_duplicate_key:
        return _rejected("duplicate_yaml_key", "A YAML mapping key is declared more than once.")
    try:
        data = yaml.safe_load(decoded)
        rule = Rule.model_validate_json(json.dumps(data))
    except RecursionError:
        return _rejected("yaml_nesting_limit", "The rule file exceeds the YAML nesting limit.")
    except (TypeError, ValidationError, yaml.YAMLError):
        return _rejected("invalid_rule_shape", "The rule does not match the version-one schema.")
    return RuleLoadResult(rule=rule)


def _has_duplicate_key(node: Node) -> bool:
    """Reject duplicate keys at every YAML mapping level before value construction."""
    if isinstance(node, MappingNode):
        keys: set[tuple[str, str]] = set()
        for key, value in node.value:
            identifier = (key.tag, str(key.value))
            if identifier in keys:
                return True
            keys.add(identifier)
            if _has_duplicate_key(key) or _has_duplicate_key(value):
                return True
    elif (
        hasattr(node, "value")
        and isinstance(node.value, Iterable)
        and not isinstance(node.value, str)
    ):
        return any(_has_duplicate_key(item) for item in node.value)
    return False


def _exceeds_flow_nesting(tokens: tuple[object, ...]) -> bool:
    """Reject deeply nested flow collections before YAML node construction."""
    depth = 0
    for token in tokens:
        if isinstance(token, (FlowMappingStartToken, FlowSequenceStartToken)):
            depth += 1
            if depth > MAX_YAML_NESTING:
                return True
        elif isinstance(token, (FlowMappingEndToken, FlowSequenceEndToken)):
            depth -= 1
    return False


def _rejected(code: str, message: str) -> RuleLoadResult:
    return RuleLoadResult(rule=None, diagnostics=(RuleDiagnostic(code=code, message=message),))
