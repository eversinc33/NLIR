"""Pure evaluation of closed rules over canonical behavior graphs."""

from __future__ import annotations

from itertools import product

from nlir.artifacts.models import SourceArtifact
from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalFragment,
    CanonicalOperation,
    CanonicalRelationship,
)
from nlir.contracts.ir import Polarity, RelationType
from nlir.graph.behavior import BehaviorGraph
from nlir.graph.models import ModalityFilter
from nlir.rules.models import (
    AnySelector,
    DecodedFromCondition,
    DirectCondition,
    DistanceCondition,
    MatchedRecord,
    ModalityCondition,
    OperationSelector,
    OperationUsesCondition,
    PathCondition,
    PathKind,
    Rule,
    RuleCondition,
    RuleResult,
    SequenceCondition,
    TrustBoundaryCondition,
)

type CanonicalRecord = CanonicalEntity | CanonicalOperation | CanonicalRelationship
type SelectorRecord = CanonicalEntity | CanonicalOperation


def evaluate_rule(
    rule: Rule, fragment: CanonicalFragment, source: SourceArtifact | None = None
) -> RuleResult:
    """Evaluate one validated rule against one complete canonical fragment.

    The function reads no raw source text. It uses only the graph API and
    canonical records that belong to a complete match. A decoded-from condition
    also reads the stored source artifact provenance, never raw source text.

    This function does not limit graph size or search work. A caller that
    receives fragments of unpredictable size must bound that size itself,
    before calling this function.
    """
    if not isinstance(rule, Rule):
        raise TypeError("evaluate_rule requires a validated Rule")
    if not isinstance(fragment, CanonicalFragment):
        raise TypeError("evaluate_rule requires a complete CanonicalFragment")
    if source is not None and source.artifact_id != fragment.artifact_id:
        raise ValueError("evaluate_rule source must belong to the canonical fragment")

    graph = BehaviorGraph(fragment)
    selector_names = tuple(sorted(rule.select))
    candidates = {name: _selector_candidates(graph, rule.select[name]) for name in selector_names}
    if any(not records for records in candidates.values()):
        return RuleResult(status="NO_HIT")

    for records in product(*(candidates[name] for name in selector_names)):
        selected = dict(zip(selector_names, records, strict=True))
        matched_relationships: list[CanonicalRelationship] = []
        matched_operations: list[CanonicalOperation] = [
            record for record in records if isinstance(record, CanonicalOperation)
        ]
        complete_match = True
        for condition in rule.where:
            condition_match, relationships, operations = _condition_matches(
                graph, condition, selected, source
            )
            if not condition_match:
                complete_match = False
                break
            matched_relationships.extend(relationships)
            matched_operations.extend(operations)
        if complete_match:
            return _hit(rule, selected, matched_relationships, matched_operations)
    return RuleResult(status="NO_HIT")


def _selector_candidates(graph: BehaviorGraph, selector: object) -> tuple[SelectorRecord, ...]:
    """Resolve one typed selector through exact graph queries."""
    if isinstance(selector, AnySelector):
        records = tuple(
            record
            for variant in selector.any
            for record in _selector_candidates(graph, variant)
        )
        return _unique_records(records)
    if isinstance(selector, OperationSelector):
        predicate = selector.operation
        modality_filter = _modality_filter(predicate.model_dump(exclude={"op"}, exclude_none=True))
        return tuple(
            operation
            for operation in graph.operations(modality_filter).records
            if predicate.op is None or operation.op is predicate.op
        )
    return graph.entities(selector.entity).records


def _modality_filter(values: dict[str, object]) -> ModalityFilter:
    """Convert rule literals into the strict canonical modality enum."""
    polarity = values.get("polarity")
    if polarity is not None:
        values = {**values, "polarity": Polarity(polarity)}
    return ModalityFilter(**values)


def _condition_matches(
    graph: BehaviorGraph,
    condition: RuleCondition,
    selected: dict[str, SelectorRecord],
    source_artifact: SourceArtifact | None,
) -> tuple[bool, tuple[CanonicalRelationship, ...], tuple[CanonicalOperation, ...]]:
    """Evaluate one declared condition through BehaviorGraph methods."""
    if isinstance(condition, DirectCondition):
        records = graph.direct_relationships(
            source=_entity_id(selected, condition.direct.from_),
            relation=condition.direct.relation,
            target=_entity_id(selected, condition.direct.to),
        ).records
        return bool(records), records, ()
    if isinstance(condition, TrustBoundaryCondition):
        source = _entity_id(selected, condition.trust_boundary.from_)
        target = _entity_id(selected, condition.trust_boundary.to)
        records = tuple(
            record
            for record in graph.trust_boundary_relationships().records
            if record.source == source and record.target == target
        )
        return bool(records), records, ()
    if isinstance(condition, (PathCondition, DistanceCondition)):
        if isinstance(condition, PathCondition):
            spec = condition.path
            max_depth = None
        else:
            spec = condition.distance
            max_depth = spec.max_depth
        source_id = _entity_id(selected, spec.from_)
        target = _entity_id(selected, spec.to)
        if spec.kind is PathKind.DERIVATION:
            result = graph.derivation_paths(source_id, target, max_depth)
        else:
            result = graph.paths(source_id, target, max_depth)
        records = result.paths[0].relationships if result.paths else ()
        return bool(result.paths), records, ()
    if isinstance(condition, SequenceCondition):
        result = graph.operation_sequences(
            _operation_id(selected, condition.sequence.from_),
            _operation_id(selected, condition.sequence.to),
        )
        records = result.sequences[0].operations if result.sequences else ()
        return bool(result.sequences), (), records
    if isinstance(condition, ModalityCondition):
        record = selected[condition.modality.selector]
        filters = _modality_filter(
            condition.modality.model_dump(exclude={"selector"}, exclude_none=True)
        )
        matching_records = graph.operations(filters).records
        return isinstance(record, CanonicalOperation) and record in matching_records, (), ()
    if isinstance(condition, OperationUsesCondition):
        operation = _operation(selected, condition.uses.operation)
        entity_id = _entity_id(selected, condition.uses.entity)
        if condition.uses.role == "actor":
            references = (operation.actor,)
        elif condition.uses.role == "input":
            references = operation.inputs
        elif condition.uses.role == "output":
            references = operation.outputs
        elif condition.uses.role == "destination":
            references = (operation.destination,)
        else:
            references = (
                operation.actor,
                *operation.inputs,
                *operation.outputs,
                operation.destination,
            )
        if entity_id in references:
            return True, (), (operation,)
        # A reference may name a still-encoded blob whose one decoded resolution
        # is this entity, linked by a DECODES_TO relationship built deterministically
        # from decode provenance (see _resolve_decoded_references).
        decodes_to = tuple(
            relationship
            for reference in references
            if reference is not None
            for relationship in graph.direct_relationships(
                source=reference, relation=RelationType.DECODES_TO, target=entity_id
            ).records
        )
        return bool(decodes_to), decodes_to, (operation,)
    if isinstance(condition, DecodedFromCondition):
        codec = condition.decoded_from.codec
        return (
            source_artifact is not None
            and source_artifact.decode_provenance is not None
            and (codec is None or source_artifact.decode_provenance.codec is codec),
            (),
            (),
        )
    raise TypeError(f"unsupported rule condition: {type(condition).__name__}")


def _entity_id(selected: dict[str, SelectorRecord], name: str) -> str:
    """Get one entity selector ID after rule validation has checked its type."""
    record = selected[name]
    if not isinstance(record, CanonicalEntity):
        raise TypeError(f"selector {name!r} is not an entity")
    return record.id


def _operation_id(selected: dict[str, SelectorRecord], name: str) -> str:
    """Get one operation selector ID after rule validation has checked its type."""
    return _operation(selected, name).id


def _operation(selected: dict[str, SelectorRecord], name: str) -> CanonicalOperation:
    """Get one operation selector after rule validation has checked its type."""
    record = selected[name]
    if not isinstance(record, CanonicalOperation):
        raise TypeError(f"selector {name!r} is not an operation")
    return record


def _hit(
    rule: Rule,
    selected: dict[str, SelectorRecord],
    relationships: list[CanonicalRelationship],
    operations: list[CanonicalOperation],
) -> RuleResult:
    """Build a stable evidence-backed binary hit."""
    entities = [record for record in selected.values() if isinstance(record, CanonicalEntity)]
    all_operations = [
        *[record for record in selected.values() if isinstance(record, CanonicalOperation)],
        *operations,
    ]
    records = _unique_records((*entities, *all_operations, *relationships))
    explanation = f"{rule.id}: {', '.join(_condition_name(condition) for condition in rule.where)}"
    return RuleResult(
        status="HIT",
        matched_entity_ids=tuple(sorted(record.id for record in entities)),
        matched_operation_ids=tuple(
            sorted(record.id for record in _unique_records(all_operations))
        ),
        matched_relationship_ids=tuple(
            sorted(record.id for record in _unique_records(relationships))
        ),
        evidence=tuple(_matched_record(record) for record in records),
        explanation=explanation,
    )


def _unique_records(
    records: tuple[CanonicalRecord, ...] | list[CanonicalRecord],
) -> tuple[CanonicalRecord, ...]:
    """Remove repeated canonical records and sort their exact source evidence."""
    distinct = {record.id: record for record in records}
    return tuple(sorted(distinct.values(), key=_record_evidence_key))


def _record_evidence_key(record: CanonicalRecord) -> tuple[str, int, int, str]:
    """Sort records by their first exact source span and canonical ID."""
    first_span = min(record.evidence, key=lambda span: (span.artifact_id, span.start, span.end))
    return (first_span.artifact_id, first_span.start, first_span.end, record.id)


def _matched_record(record: CanonicalRecord) -> MatchedRecord:
    """Keep only exact canonical evidence, in a stable span order."""
    if isinstance(record, CanonicalEntity):
        record_type = "entity"
    elif isinstance(record, CanonicalOperation):
        record_type = "operation"
    else:
        record_type = "relationship"
    return MatchedRecord(
        record_id=record.id,
        record_type=record_type,
        spans=tuple(
            sorted(record.evidence, key=lambda span: (span.artifact_id, span.start, span.end))
        ),
    )


def _condition_name(condition: RuleCondition) -> str:
    """Return the declared condition name for a concise stable explanation."""
    if isinstance(condition, DirectCondition):
        return "direct"
    if isinstance(condition, TrustBoundaryCondition):
        return "trust_boundary"
    if isinstance(condition, PathCondition):
        return "path"
    if isinstance(condition, SequenceCondition):
        return "sequence"
    if isinstance(condition, ModalityCondition):
        return "modality"
    if isinstance(condition, OperationUsesCondition):
        return "uses"
    if isinstance(condition, DistanceCondition):
        return "distance"
    if isinstance(condition, DecodedFromCondition):
        return "decoded_from"
    raise TypeError(f"unsupported rule condition: {type(condition).__name__}")
