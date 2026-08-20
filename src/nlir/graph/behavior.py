"""Pure immutable indexes and queries over one canonical artifact."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from types import MappingProxyType

from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalFragment,
    CanonicalOperation,
    CanonicalRelationship,
)
from nlir.contracts.ir import RelationType
from nlir.graph.models import (
    EntityFilter,
    EntityQueryResult,
    GraphPath,
    ModalityFilter,
    OperationQueryResult,
    OperationSequence,
    PathQueryResult,
    RelationshipQueryResult,
    SequenceQueryResult,
)


class BehaviorGraph:
    """An immutable, one-artifact view of declared canonical behavior only."""

    def __init__(self, fragment: CanonicalFragment) -> None:
        if not isinstance(fragment, CanonicalFragment):
            raise TypeError("BehaviorGraph requires a CanonicalFragment")
        self._fragment = fragment
        entities = {entity.id: entity for entity in fragment.entities}
        operations = {operation.id: operation for operation in fragment.operations}
        relationships = {relationship.id: relationship for relationship in fragment.relationships}
        if len(entities) != len(fragment.entities):
            raise ValueError("canonical fragment contains duplicate entity IDs")
        if len(operations) != len(fragment.operations):
            raise ValueError("canonical fragment contains duplicate operation IDs")
        if len(relationships) != len(fragment.relationships):
            raise ValueError("canonical fragment contains duplicate relationship IDs")
        self._entities = MappingProxyType(entities)
        self._operations = MappingProxyType(operations)
        self._relationships = MappingProxyType(relationships)
        self._validate_references()
        self._sorted_entities = tuple(sorted(self._entities.values(), key=lambda record: record.id))
        self._sorted_operations = tuple(
            sorted(self._operations.values(), key=lambda record: record.id)
        )
        self._sorted_relationships = tuple(
            sorted(self._relationships.values(), key=lambda record: record.id)
        )
        adjacency: dict[str, list[CanonicalRelationship]] = defaultdict(list)
        for relationship in self._sorted_relationships:
            adjacency[relationship.source].append(relationship)
        self._adjacency = MappingProxyType(
            {
                source: tuple(sorted(edges, key=lambda edge: (edge.target, edge.id)))
                for source, edges in adjacency.items()
            }
        )
        producers: dict[str, list[CanonicalOperation]] = defaultdict(list)
        consumers: dict[str, list[CanonicalOperation]] = defaultdict(list)
        for operation in self._sorted_operations:
            for output in operation.outputs:
                producers[output].append(operation)
            for input_id in operation.inputs:
                consumers[input_id].append(operation)
        self._producers = MappingProxyType(
            {
                entity_id: tuple(sorted(records, key=lambda record: record.id))
                for entity_id, records in producers.items()
            }
        )
        self._consumers = MappingProxyType(
            {
                entity_id: tuple(sorted(records, key=lambda record: record.id))
                for entity_id, records in consumers.items()
            }
        )
        self._operation_adjacency = MappingProxyType(
            {
                operation.id: tuple(
                    sorted(
                        {
                            consumer.id: consumer
                            for output in operation.outputs
                            for consumer in self._consumers.get(output, ())
                        }.values(),
                        key=lambda candidate: candidate.id,
                    )
                )
                for operation in self._sorted_operations
            }
        )

    def _validate_references(self) -> None:
        for relationship in self._relationships.values():
            self._require_entity(relationship.source, "relationship source")
            self._require_entity(relationship.target, "relationship target")
        for operation in self._operations.values():
            for identifier, label in (
                *((input_id, "operation input") for input_id in operation.inputs),
                *((output, "operation output") for output in operation.outputs),
            ):
                self._require_entity(identifier, label)
            if operation.actor is not None:
                self._require_entity(operation.actor, "operation actor")
            if operation.destination is not None:
                self._require_entity(operation.destination, "operation destination")

    def _require_entity(self, identifier: str, label: str) -> None:
        if identifier not in self._entities:
            raise ValueError(f"{label} references unknown entity: {identifier}")

    def direct_relationships(
        self,
        *,
        source: str | None = None,
        relation: RelationType | None = None,
        target: str | None = None,
    ) -> RelationshipQueryResult:
        """Return declared relationships matching all supplied exact filters."""
        return RelationshipQueryResult(
            records=tuple(
                item
                for item in self._sorted_relationships
                if (source is None or item.source == source)
                and (relation is None or item.relation is relation)
                and (target is None or item.target == target)
            )
        )

    def entities(self, filters: EntityFilter = EntityFilter()) -> EntityQueryResult:
        """Return entities matching every declared predicate.

        Every field is exact equality except ``value_pattern``, which searches
        the entity's ``value`` for a regular expression instead of comparing it.
        """
        return EntityQueryResult(
            records=tuple(
                entity for entity in self._sorted_entities if self._entity_matches(entity, filters)
            )
        )

    def _entity_matches(self, entity: CanonicalEntity, filters: EntityFilter) -> bool:
        equality_fields = filters.model_dump(exclude={"value_pattern"})
        if any(
            value is not None and getattr(entity, field) != value
            for field, value in equality_fields.items()
        ):
            return False
        if filters.value_pattern is not None:
            if entity.value is None or not re.search(filters.value_pattern, entity.value):
                return False
        return True

    def operations(self, filters: ModalityFilter = ModalityFilter()) -> OperationQueryResult:
        """Return operations matching only explicitly declared modality fields."""
        return OperationQueryResult(
            records=tuple(
                operation
                for operation in self._sorted_operations
                if all(
                    value is None or getattr(operation.modality, field) == value
                    for field, value in filters.model_dump().items()
                )
            )
        )

    def trust_boundary_relationships(self) -> RelationshipQueryResult:
        """Return declared relationships whose declared endpoint trusts differ."""
        return RelationshipQueryResult(
            records=tuple(
                relationship
                for relationship in self._sorted_relationships
                if self._entities[relationship.source].trust
                != self._entities[relationship.target].trust
            )
        )

    def paths(self, source: str, sink: str, max_depth: int | None = None) -> PathQueryResult:
        """Find source-to-sink paths over declared relationship direction.

        ``max_depth`` caps the number of relationship hops a path may use. It is
        the caller's own semantic parameter, not a general safety limit; pass
        ``None`` for an unbounded search.
        """
        return self._relationship_paths(source, sink, lambda _edge: True, max_depth)

    def derivation_paths(
        self, source: str, sink: str, max_depth: int | None = None
    ) -> PathQueryResult:
        """Find paths using only declared ``DERIVED_FROM`` relationships.

        See ``paths`` for the meaning of ``max_depth``.
        """
        return self._relationship_paths(
            source, sink, lambda edge: edge.relation is RelationType.DERIVED_FROM, max_depth
        )

    def _relationship_paths(
        self,
        source: str,
        sink: str,
        include: Callable[[CanonicalRelationship], bool],
        max_depth: int | None,
    ) -> PathQueryResult:
        if source not in self._entities or sink not in self._entities:
            return PathQueryResult()
        paths: list[GraphPath] = []
        stack: list[tuple[str, tuple[str, ...], tuple[CanonicalRelationship, ...]]] = [
            (source, (source,), ())
        ]
        while stack:
            node, node_ids, edge_records = stack.pop()
            if node == sink:
                paths.append(GraphPath(entity_ids=node_ids, relationships=edge_records))
                continue
            if max_depth is not None and len(edge_records) >= max_depth:
                continue
            edges = tuple(edge for edge in self._adjacency.get(node, ()) if include(edge))
            for edge in reversed(edges):
                if edge.target in node_ids:
                    continue
                stack.append((edge.target, (*node_ids, edge.target), (*edge_records, edge)))
        ordered_paths = tuple(sorted(paths, key=lambda path: path.entity_ids))
        return PathQueryResult(paths=ordered_paths)

    def operation_sequences(self, source: str, sink: str) -> SequenceQueryResult:
        """Find sequences connected only by declared output-to-input equality."""
        if source not in self._operations or sink not in self._operations:
            return SequenceQueryResult()
        sequences: list[OperationSequence] = []
        stack: list[tuple[CanonicalOperation, tuple[CanonicalOperation, ...]]] = [
            (self._operations[source], (self._operations[source],))
        ]
        while stack:
            operation, records = stack.pop()
            if operation.id == sink:
                sequences.append(
                    OperationSequence(
                        operation_ids=tuple(record.id for record in records), operations=records
                    )
                )
                continue
            candidates = self._operation_adjacency[operation.id]
            seen = {record.id for record in records}
            for candidate in reversed(candidates):
                if candidate.id not in seen:
                    stack.append((candidate, (*records, candidate)))
        ordered = tuple(sorted(sequences, key=lambda sequence: sequence.operation_ids))
        return SequenceQueryResult(sequences=ordered)
