"""Core analysis logic for the browser view, built entirely on the nlir library.

The browser view lifts through the live model only. There is no offline or
fixture lifting path here: storage and reproducibility of a lift are the
caller's responsibility, not the browser view's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nlir.artifacts.loader import LoadedArtifact
from nlir.artifacts.models import SourceArtifact
from nlir.canonical.models import CanonicalFragment
from nlir.contracts.common import SourceSpan
from nlir.ir.models import ArtifactRecord
from nlir.ir.service import lift_loaded_artifact, resolve_input_source_span
from nlir.lifting.live import LiveResponsesLifter
from nlir.lifting.models import CanonicalAttemptStage
from nlir.rules.evaluate import evaluate_rule
from nlir.rules.loader import load_rule
from nlir.rules.models import Rule

MAX_PROMPT_CHARS = 128 * 1024
"""Bound interactive source text before local processing begins."""


class ViewerInputError(ValueError):
    """Reject one browser request with a safe, user-facing message."""


@dataclass(frozen=True)
class RuleDocument:
    """One readable, validated rule file for the browser view."""

    path: Path
    text: str
    rule: Rule


@dataclass(frozen=True)
class _Fragment:
    """One accepted canonical fragment, tagged with a globally unique node prefix."""

    prefix: str
    artifact_id: str
    ordinal: int
    fragment: CanonicalFragment


class Inspector:
    """Read local rules and lift one browser-provided prompt through the live model."""

    def __init__(self, *, lifter: LiveResponsesLifter, metadata, rules_directory: Path) -> None:
        if not rules_directory.is_dir():
            raise ValueError("The rules directory is not available.")
        self._lifter = lifter
        self._metadata = metadata
        self._rules_directory = rules_directory

    @classmethod
    def from_live_config(cls, config_path: Path, rules_directory: Path) -> Inspector:
        """Create an inspector from the explicit non-secret live configuration."""
        lifter = LiveResponsesLifter.from_toml_file(config_path)
        metadata = lifter.lift_metadata()
        if metadata is None:
            raise ValueError("The live configuration is invalid.")
        return cls(lifter=lifter, metadata=metadata, rules_directory=rules_directory)

    def list_rules(self) -> list[dict[str, Any]]:
        """Return every readable valid rule without sending its source text."""
        return [
            {
                "id": document.rule.id,
                "description": _rule_description(document.rule),
                "author": document.rule.metadata.author if document.rule.metadata else None,
            }
            for document in self._rule_documents()
        ]

    def rule_detail(self, rule_id: str) -> dict[str, Any]:
        """Return one local rule in readable text form."""
        for document in self._rule_documents():
            if document.rule.id == rule_id:
                return {
                    "id": document.rule.id,
                    "text": document.text,
                    "description": _rule_description(document.rule),
                    "author": document.rule.metadata.author if document.rule.metadata else None,
                    "references": list(document.rule.metadata.references)
                    if document.rule.metadata
                    else [],
                }
        raise ViewerInputError("The selected rule is not available.")

    def analyze(self, prompt: str) -> dict[str, Any]:
        """Lift one prompt through the live model and evaluate all local rules."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ViewerInputError("Enter prompt text before analysis.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ViewerInputError("The prompt exceeds the 128 KiB limit.")
        root = SourceArtifact.from_text(prompt, source_name="viewer-input")
        records = lift_loaded_artifact(
            LoadedArtifact(artifact=root, relative_path="viewer-input"),
            lifter=self._lifter,
            metadata=self._metadata,
        )
        sources = {record.source.artifact_id: record.source for record in records}
        fragments = _accepted_fragments(records)

        nodes, edges = _graph_elements(fragments)
        return {
            "tokens": _text_tokens(fragments, sources, root.artifact_id),
            "graph": {"nodes": nodes, "edges": edges},
            "rules": [
                _rule_match(document, fragments, sources, root.artifact_id)
                for document in self._rule_documents()
            ],
            "diagnostics": _diagnostics(records),
            "attempts": _attempts(records),
        }

    def _rule_documents(self) -> tuple[RuleDocument, ...]:
        documents: list[RuleDocument] = []
        for path in sorted(self._rules_directory.glob("*.yaml")):
            loaded = load_rule(path)
            if loaded.rule is None:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            documents.append(RuleDocument(path=path, text=text, rule=loaded.rule))
        return tuple(documents)


def _rule_description(rule: Rule) -> str | None:
    """Return the one human description when a rule has one."""
    return rule.metadata.description if rule.metadata else rule.description


def _accepted_fragments(records: tuple[ArtifactRecord, ...]) -> tuple[_Fragment, ...]:
    """Collect every accepted canonical fragment with a globally unique node prefix."""
    fragments: list[_Fragment] = []
    for record in records:
        for attempt in record.canonical_attempts:
            if attempt.stage is not CanonicalAttemptStage.ACCEPTED:
                continue
            if attempt.canonical_fragment is None:
                continue
            fragments.append(
                _Fragment(
                    prefix=f"{record.source.artifact_id}:{attempt.ordinal}",
                    artifact_id=record.source.artifact_id,
                    ordinal=attempt.ordinal,
                    fragment=attempt.canonical_fragment,
                )
            )
    return tuple(fragments)


def _root_spans(
    evidence: tuple[SourceSpan, ...],
    sources: dict[str, SourceArtifact],
    root_artifact_id: str,
) -> list[dict[str, int]]:
    """Map decoded evidence back onto the one prompt the operator pasted."""
    spans = []
    for span in evidence:
        resolved = resolve_input_source_span(span, sources)
        if resolved.artifact_id == root_artifact_id:
            spans.append({"start": resolved.start, "end": resolved.end})
    return spans


def _graph_elements(fragments: tuple[_Fragment, ...]) -> tuple[list[dict], list[dict]]:
    """Render every accepted fragment as a behavior graph of nodes and role edges."""
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    for item in fragments:
        for entity in item.fragment.entities:
            nodes.append(
                {
                    "data": {
                        "id": f"{item.prefix}:{entity.id}",
                        "label": entity.value or entity.type.value,
                        "kind": "entity",
                        "type": entity.type.value,
                        "sensitivity": entity.sensitivity.value,
                        "trust": entity.trust.value,
                    }
                }
            )
        for operation in item.fragment.operations:
            node_id = f"{item.prefix}:{operation.id}"
            nodes.append(
                {
                    "data": {
                        "id": node_id,
                        "label": operation.op.value,
                        "kind": "operation",
                        "type": operation.op.value,
                    }
                }
            )
            for input_id in operation.inputs:
                edges.append(_role_edge(f"{item.prefix}:{input_id}", node_id, "input"))
            for output_id in operation.outputs:
                edges.append(_role_edge(node_id, f"{item.prefix}:{output_id}", "output"))
            if operation.actor is not None:
                edges.append(_role_edge(f"{item.prefix}:{operation.actor}", node_id, "actor"))
            if operation.destination is not None:
                edges.append(
                    _role_edge(node_id, f"{item.prefix}:{operation.destination}", "destination")
                )
        for relationship in item.fragment.relationships:
            edges.append(
                {
                    "data": {
                        "id": f"{item.prefix}:{relationship.id}",
                        "source": f"{item.prefix}:{relationship.source}",
                        "target": f"{item.prefix}:{relationship.target}",
                        "label": relationship.relation.value,
                        "kind": "relationship",
                        "type": relationship.relation.value,
                    }
                }
            )
    return nodes, edges


def _role_edge(source: str, target: str, role: str) -> dict[str, Any]:
    """Build one operation-to-entity role edge (input, output, actor, destination)."""
    return {
        "data": {
            "id": f"{source}->{target}:{role}",
            "source": source,
            "target": target,
            "label": role,
            "kind": "role",
            "type": role,
        }
    }


def _text_tokens(
    fragments: tuple[_Fragment, ...],
    sources: dict[str, SourceArtifact],
    root_artifact_id: str,
) -> list[dict[str, Any]]:
    """Flatten every accepted fact into prompt-relative spans for text highlighting."""
    tokens: list[dict[str, Any]] = []
    for item in fragments:
        for entity in item.fragment.entities:
            node_id = f"{item.prefix}:{entity.id}"
            for span in _root_spans(entity.evidence, sources, root_artifact_id):
                tokens.append(
                    {"node_id": node_id, "kind": "entity", "type": entity.type.value, **span}
                )
        for operation in item.fragment.operations:
            node_id = f"{item.prefix}:{operation.id}"
            for span in _root_spans(operation.evidence, sources, root_artifact_id):
                tokens.append(
                    {"node_id": node_id, "kind": "operation", "type": operation.op.value, **span}
                )
        for relationship in item.fragment.relationships:
            node_id = f"{item.prefix}:{relationship.id}"
            for span in _root_spans(relationship.evidence, sources, root_artifact_id):
                tokens.append(
                    {
                        "node_id": node_id,
                        "kind": "relationship",
                        "type": relationship.relation.value,
                        **span,
                    }
                )
    tokens.sort(key=lambda token: (token["start"], token["end"], token["node_id"]))
    return tokens


def _rule_match(
    document: RuleDocument,
    fragments: tuple[_Fragment, ...],
    sources: dict[str, SourceArtifact],
    root_artifact_id: str,
) -> dict[str, Any]:
    """Evaluate one rule against every accepted fragment and render matched node IDs."""
    matches: list[dict[str, Any]] = []
    for item in fragments:
        result = evaluate_rule(document.rule, item.fragment, sources[item.artifact_id])
        if result.status != "HIT":
            continue
        matched_ids = (
            *result.matched_entity_ids,
            *result.matched_operation_ids,
            *result.matched_relationship_ids,
        )
        spans: list[dict[str, int]] = []
        for matched_record in result.evidence:
            spans.extend(_root_spans(matched_record.spans, sources, root_artifact_id))
        matches.append(
            {
                "artifact_id": item.artifact_id,
                "attempt_ordinal": item.ordinal,
                "matched_node_ids": [f"{item.prefix}:{identifier}" for identifier in matched_ids],
                "explanation": result.explanation,
                "spans": spans,
            }
        )
    return {
        "id": document.rule.id,
        "status": "HIT" if matches else "NO_HIT",
        "matches": matches,
    }


def _diagnostics(records: tuple[ArtifactRecord, ...]) -> list[str]:
    """Return unique non-finding diagnostic codes in stable order."""
    return sorted(
        {diagnostic.code for record in records for diagnostic in record.decode_diagnostics}
    )


def _attempts(records: tuple[ArtifactRecord, ...]) -> list[dict[str, Any]]:
    """Return accepted or rejected attempt states without provider response data."""
    return [
        {
            "artifact_id": record.source.artifact_id,
            "ordinal": attempt.ordinal,
            "state": attempt.stage.value,
            "diagnostics": [diagnostic.code for diagnostic in attempt.diagnostics],
        }
        for record in records
        for attempt in record.canonical_attempts
    ]
