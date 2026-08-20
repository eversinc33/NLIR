"""Pure, atomic conversion from validated source semantics to canonical facts."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalFragment,
    CanonicalOperation,
    CanonicalRelationship,
    NormalizationDiagnostic,
    NormalizationResult,
    SourceCanonicalId,
)
from nlir.contracts.common import SourceSpan
from nlir.contracts.diagnostics import DiagnosticSeverity
from nlir.contracts.ir import Entity, IRFragment, Operation, Relationship
from nlir.contracts.validation import validate_fragment

if TYPE_CHECKING:
    from nlir.artifacts.models import SourceArtifact


def normalize_fragment(
    fragment: IRFragment,
    artifact_id: str,
    artifacts: Mapping[str, SourceArtifact],
) -> NormalizationResult:
    """Canonically reconcile one accepted artifact fragment without repairing it.

    The Phase 1 validation boundary is deliberately re-run here, then all facts
    are prepared locally.  A disagreement anywhere discards the entire candidate.
    """
    validated = validate_fragment(fragment, artifacts)
    if validated.fragment is None:
        return _rejected(
            tuple(
                _diagnostic(diagnostic.code, diagnostic.message, diagnostic.span)
                for diagnostic in validated.diagnostics
            )
        )
    if artifact_id not in artifacts:
        return _rejected(
            (
                _diagnostic(
                    "target_artifact_not_found", "Target artifact is not in the source registry."
                ),
            )
        )
    cross_artifact_spans = _cross_artifact_spans(validated.fragment, artifact_id)
    if cross_artifact_spans:
        return _rejected(
            tuple(
                _diagnostic(
                    "cross_artifact_evidence",
                    "Canonicalization accepts evidence only from its target artifact.",
                    source_span,
                )
                for source_span in cross_artifact_spans
            )
        )

    entity_conflicts = _entity_conflicts(validated.fragment.entities)
    if entity_conflicts:
        return _rejected(entity_conflicts)

    entities, entity_mapping = _canonical_entities(validated.fragment.entities, artifact_id)
    operations, operation_mapping, operation_conflicts = _canonical_operations(
        validated.fragment.operations, entity_mapping, artifact_id
    )
    if operation_conflicts:
        return _rejected(operation_conflicts)
    relationships, relationship_conflicts = _canonical_relationships(
        validated.fragment.relationships, entity_mapping, artifact_id
    )
    if relationship_conflicts:
        return _rejected(relationship_conflicts)

    mappings = tuple(
        SourceCanonicalId(source_id=source_id, canonical_id=canonical_id)
        for source_id, canonical_id in sorted(
            {**entity_mapping, **operation_mapping}.items(), key=lambda item: item[0]
        )
    )
    return NormalizationResult(
        fragment=CanonicalFragment(
            artifact_id=artifact_id,
            entities=tuple(sorted(entities, key=lambda item: item.id)),
            operations=tuple(sorted(operations, key=lambda item: item.id)),
            relationships=tuple(sorted(relationships, key=lambda item: item.id)),
        ),
        source_to_canonical=mappings,
    )


def _entity_conflicts(entities: Sequence[Entity]) -> tuple[NormalizationDiagnostic, ...]:
    groups: dict[tuple[object, object], list[Entity]] = defaultdict(list)
    for entity in entities:
        if entity.value is not None:
            groups[(entity.type, entity.value)].append(entity)
    diagnostics: list[NormalizationDiagnostic] = []
    for group in groups.values():
        properties = {
            (item.subtype, item.sensitivity, item.trust, item.confidence, item.underspecified)
            for item in group
        }
        if len(properties) > 1:
            diagnostics.append(
                _diagnostic(
                    "entity_reconciliation_conflict",
                    "Entities with the same type and value disagree on canonical properties.",
                    _first_span(group),
                )
            )
    return _sorted_diagnostics(diagnostics)


def _canonical_entities(
    entities: Sequence[Entity], artifact_id: str
) -> tuple[list[CanonicalEntity], dict[str, str]]:
    grouped: dict[tuple[object, ...], list[Entity]] = defaultdict(list)
    for entity in entities:
        if entity.value is None:
            # Value-less facts have no declared canonical identity and must remain distinct.
            key = ("unvalued", entity.id)
        else:
            key = ("valued", entity.type, entity.value)
        grouped[key].append(entity)

    canonical_entities: list[CanonicalEntity] = []
    mapping: dict[str, str] = {}
    for key, group in grouped.items():
        representative = group[0]
        identity: dict[str, object] = {
            "artifact_id": artifact_id,
            "type": representative.type.value,
            "value": representative.value,
        }
        if representative.value is None:
            identity["source_instance"] = representative.id
        canonical_id = _canonical_id(
            "entity", representative.type.value, representative.value, identity
        )
        canonical_entities.append(
            CanonicalEntity(
                id=canonical_id,
                type=representative.type,
                subtype=representative.subtype,
                value=representative.value,
                sensitivity=representative.sensitivity,
                trust=representative.trust,
                evidence=_evidence_union(*(item.evidence for item in group)),
                confidence=representative.confidence,
                underspecified=representative.underspecified,
            )
        )
        mapping.update({item.id: canonical_id for item in group})
    return canonical_entities, mapping


def _canonical_operations(
    operations: Sequence[Operation],
    entity_mapping: Mapping[str, str],
    artifact_id: str,
) -> tuple[list[CanonicalOperation], dict[str, str], tuple[NormalizationDiagnostic, ...]]:
    remapped: list[tuple[Operation, tuple[object, ...], tuple[object, ...]]] = []
    for operation in operations:
        semantic = (
            operation.op,
            _mapped_optional(operation.actor, entity_mapping),
            tuple(entity_mapping[identifier] for identifier in operation.inputs),
            tuple(entity_mapping[identifier] for identifier in operation.outputs),
            _mapped_optional(operation.destination, entity_mapping),
        )
        full = (*semantic, operation.modality, operation.confidence, operation.underspecified)
        remapped.append((operation, semantic, full))
    conflicts = _conflicts_for_records(
        remapped,
        code="operation_reconciliation_conflict",
        message="Operations with matching endpoints disagree on modality or canonical properties.",
    )
    if conflicts:
        return [], {}, conflicts

    groups: dict[tuple[object, ...], list[tuple[Operation, tuple[object, ...]]]] = defaultdict(list)
    for operation, _semantic, full in remapped:
        groups[full].append((operation, full))
    canonical_operations: list[CanonicalOperation] = []
    mapping: dict[str, str] = {}
    for full, group in groups.items():
        operation = group[0][0]
        op, actor, inputs, outputs, destination, modality, confidence, underspecified = full
        identity = {
            "artifact_id": artifact_id,
            "op": op.value,
            "actor": actor,
            "inputs": inputs,
            "outputs": outputs,
            "destination": destination,
            "modality": modality.model_dump(mode="json"),
            "confidence": confidence,
            "underspecified": underspecified,
        }
        canonical_id = _canonical_id("operation", op.value, None, identity)
        canonical_operations.append(
            CanonicalOperation(
                id=canonical_id,
                op=op,
                actor=actor,
                inputs=inputs,
                outputs=outputs,
                destination=destination,
                modality=modality,
                evidence=_evidence_union(*(item.evidence for item, _ in group)),
                confidence=confidence,
                underspecified=underspecified,
            )
        )
        mapping.update({item.id: canonical_id for item, _ in group})
    return canonical_operations, mapping, ()


def _canonical_relationships(
    relationships: Sequence[Relationship],
    entity_mapping: Mapping[str, str],
    artifact_id: str,
) -> tuple[list[CanonicalRelationship], tuple[NormalizationDiagnostic, ...]]:
    remapped: list[tuple[Relationship, tuple[object, ...], tuple[object, ...]]] = []
    for relationship in relationships:
        semantic = (
            entity_mapping[relationship.source],
            relationship.relation,
            entity_mapping[relationship.target],
        )
        full = (*semantic, relationship.confidence, relationship.underspecified)
        remapped.append((relationship, semantic, full))
    conflicts = _conflicts_for_records(
        remapped,
        code="relationship_reconciliation_conflict",
        message="Relationships with matching endpoints disagree on canonical properties.",
    )
    if conflicts:
        return [], conflicts

    groups: dict[tuple[object, ...], list[tuple[Relationship, tuple[object, ...]]]] = defaultdict(
        list
    )
    for relationship, _semantic, full in remapped:
        groups[full].append((relationship, full))
    canonical_relationships: list[CanonicalRelationship] = []
    for full, group in groups.items():
        source, relation, target, confidence, underspecified = full
        identity = {
            "artifact_id": artifact_id,
            "source": source,
            "relation": relation.value,
            "target": target,
            "confidence": confidence,
            "underspecified": underspecified,
        }
        canonical_id = _canonical_id("relationship", relation.value, None, identity)
        canonical_relationships.append(
            CanonicalRelationship(
                id=canonical_id,
                source=source,
                relation=relation,
                target=target,
                evidence=_evidence_union(*(item.evidence for item, _ in group)),
                confidence=confidence,
                underspecified=underspecified,
            )
        )
    return canonical_relationships, ()


def _conflicts_for_records(
    records: Sequence[tuple[object, tuple[object, ...], tuple[object, ...]]],
    *,
    code: str,
    message: str,
) -> tuple[NormalizationDiagnostic, ...]:
    by_semantic: dict[tuple[object, ...], list[tuple[object, ...]]] = defaultdict(list)
    record_by_semantic: dict[tuple[object, ...], object] = {}
    for record, semantic, full in records:
        by_semantic[semantic].append(full)
        record_by_semantic.setdefault(semantic, record)
    diagnostics: list[NormalizationDiagnostic] = []
    for semantic, full_records in by_semantic.items():
        if len(set(full_records)) > 1:
            record = record_by_semantic[semantic]
            diagnostics.append(_diagnostic(code, message, _first_span((record,))))
    return _sorted_diagnostics(diagnostics)


def _mapped_optional(identifier: str | None, mapping: Mapping[str, str]) -> str | None:
    return None if identifier is None else mapping[identifier]


def _cross_artifact_spans(fragment: IRFragment, artifact_id: str) -> tuple[SourceSpan, ...]:
    spans = [
        evidence
        for record in (*fragment.entities, *fragment.operations, *fragment.relationships)
        for evidence in record.evidence
        if evidence.artifact_id != artifact_id
    ]
    return _evidence_union(spans)


def _evidence_union(*groups: Sequence[SourceSpan]) -> tuple[SourceSpan, ...]:
    evidence = {
        (span.artifact_id, span.start, span.end): span for group in groups for span in group
    }
    return tuple(
        sorted(evidence.values(), key=lambda span: (span.artifact_id, span.start, span.end))
    )


def _first_span(records: Sequence[object]) -> SourceSpan | None:
    spans = _evidence_union(*(record.evidence for record in records))
    return spans[0] if spans else None


def _canonical_id(
    prefix: str, kind: str, readable_value: str | None, identity: Mapping[str, object]
) -> str:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    suffix = hashlib.sha256(serialized.encode("utf-8", errors="strict")).hexdigest()[:12]
    parts = [prefix, _slug(kind)]
    if readable_value is not None:
        parts.append(_slug(readable_value))
    return ".".join((*parts, suffix))


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (slug or "value")[:48]


def _diagnostic(code: str, message: str, span: SourceSpan | None = None) -> NormalizationDiagnostic:
    return NormalizationDiagnostic(
        code=code, severity=DiagnosticSeverity.ERROR, message=message, span=span
    )


def _rejected(diagnostics: tuple[NormalizationDiagnostic, ...]) -> NormalizationResult:
    return NormalizationResult(fragment=None, diagnostics=_sorted_diagnostics(diagnostics))


def _sorted_diagnostics(
    diagnostics: Sequence[NormalizationDiagnostic],
) -> tuple[NormalizationDiagnostic, ...]:
    return tuple(
        sorted(
            diagnostics,
            key=lambda item: (
                item.code,
                "" if item.span is None else item.span.artifact_id,
                -1 if item.span is None else item.span.start,
                -1 if item.span is None else item.span.end,
                item.message,
            ),
        )
    )
