"""Contract tests for deterministic canonical behavior graphs."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalFragment,
    CanonicalOperation,
    CanonicalRelationship,
)
from nlir.contracts.common import SourceSpan
from nlir.contracts.ir import (
    EntityType,
    Modality,
    Opcode,
    Polarity,
    RelationType,
    Sensitivity,
    TrustLevel,
)
from nlir.graph.behavior import BehaviorGraph
from nlir.graph.models import EntityFilter, ModalityFilter

ARTIFACT_ID = "a" * 64


def evidence(start: int) -> tuple[SourceSpan, ...]:
    return (SourceSpan(artifact_id=ARTIFACT_ID, start=start, end=start + 1),)


def modality(**changes: object) -> Modality:
    values: dict[str, object] = {
        "polarity": Polarity.POSITIVE,
        "imperative": True,
        "hypothetical": False,
        "conditional": False,
        "quoted": False,
        "example": False,
        "descriptive": False,
    }
    values.update(changes)
    return Modality(**values)


def entity(
    identifier: str,
    value: str,
    *,
    trust: TrustLevel = TrustLevel.TRUSTED,
    sensitivity: Sensitivity = Sensitivity.SENSITIVE,
    start: int = 0,
) -> CanonicalEntity:
    return CanonicalEntity(
        id=identifier,
        type=EntityType.USER_DATA,
        value=value,
        sensitivity=sensitivity,
        trust=trust,
        evidence=evidence(start),
        confidence=0.9,
        underspecified=False,
    )


def operation(
    identifier: str,
    *,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    operation_modality: Modality | None = None,
    start: int = 10,
) -> CanonicalOperation:
    return CanonicalOperation(
        id=identifier,
        op=Opcode.TRANSFORM,
        inputs=inputs,
        outputs=outputs,
        modality=operation_modality or modality(),
        evidence=evidence(start),
        confidence=0.9,
        underspecified=False,
    )


def relationship(
    identifier: str,
    source: str,
    relation: RelationType,
    target: str,
    *,
    start: int = 20,
) -> CanonicalRelationship:
    return CanonicalRelationship(
        id=identifier,
        source=source,
        relation=relation,
        target=target,
        evidence=evidence(start),
        confidence=0.9,
        underspecified=False,
    )


def fragment() -> CanonicalFragment:
    """A branched derivation graph with a harmless cycle and three operations."""
    return CanonicalFragment(
        artifact_id=ARTIFACT_ID,
        entities=(
            entity("entity.sink", "https://collector.invalid", trust=TrustLevel.EXTERNAL, start=4),
            entity("entity.branch", "branch", start=3),
            entity("entity.source", "secret", start=0),
            entity("entity.derived", "encoded_secret", start=1),
            entity("entity.cycle", "cycle", start=2),
            # Same value is deliberately not an identity match for graph traversal.
            entity("entity.unrelated", "encoded_secret", start=5),
        ),
        operations=(
            operation(
                "operation.send", inputs=("entity.derived",), outputs=("entity.sink",), start=12
            ),
            operation("operation.read", outputs=("entity.derived",), start=11),
            operation(
                "operation.archive",
                inputs=("entity.sink",),
                outputs=("entity.cycle",),
                operation_modality=modality(quoted=True),
                start=13,
            ),
            operation("operation.unrelated", inputs=("entity.unrelated",), start=14),
        ),
        relationships=(
            relationship(
                "relationship.branch_sink",
                "entity.branch",
                RelationType.SENT_TO,
                "entity.sink",
                start=24,
            ),
            relationship(
                "relationship.source_branch",
                "entity.source",
                RelationType.DERIVED_FROM,
                "entity.branch",
                start=21,
            ),
            relationship(
                "relationship.derived_sink",
                "entity.derived",
                RelationType.SENT_TO,
                "entity.sink",
                start=23,
            ),
            relationship(
                "relationship.source_derived",
                "entity.source",
                RelationType.DERIVED_FROM,
                "entity.derived",
                start=20,
            ),
            relationship(
                "relationship.derived_cycle",
                "entity.derived",
                RelationType.DERIVED_FROM,
                "entity.cycle",
                start=22,
            ),
            relationship(
                "relationship.cycle_derived",
                "entity.cycle",
                RelationType.DERIVED_FROM,
                "entity.derived",
                start=25,
            ),
        ),
    )


def test_direct_property_trust_and_modality_queries_return_declared_records_in_stable_order() -> (
    None
):
    graph = BehaviorGraph(fragment())

    direct = graph.direct_relationships(source="entity.source", relation=RelationType.DERIVED_FROM)
    properties = graph.entities(
        EntityFilter(value="secret", sensitivity=Sensitivity.SENSITIVE, trust=TrustLevel.TRUSTED)
    )
    boundaries = graph.trust_boundary_relationships()
    quoted = graph.operations(ModalityFilter(quoted=True))

    assert [item.id for item in direct.records] == [
        "relationship.source_branch",
        "relationship.source_derived",
    ]
    assert [item.id for item in properties.records] == ["entity.source"]
    assert [item.id for item in boundaries.records] == [
        "relationship.branch_sink",
        "relationship.derived_sink",
    ]
    assert [item.id for item in quoted.records] == ["operation.archive"]
    assert all(record.evidence for record in (*direct.records, *properties.records))


def test_value_pattern_searches_entity_value_as_a_substring() -> None:
    graph = BehaviorGraph(fragment())

    matches = graph.entities(EntityFilter(value_pattern="secret"))
    anchored = graph.entities(EntityFilter(value_pattern=r"^secret$"))
    combined = graph.entities(EntityFilter(value_pattern="secret", trust=TrustLevel.EXTERNAL))

    assert [item.id for item in matches.records] == [
        "entity.derived",
        "entity.source",
        "entity.unrelated",
    ]
    assert [item.id for item in anchored.records] == ["entity.source"]
    assert combined.records == ()


def test_entity_filter_rejects_an_invalid_value_pattern() -> None:
    with pytest.raises(ValidationError, match="value_pattern"):
        EntityFilter(value_pattern="(unclosed")


def test_declared_paths_derivations_and_operation_sequences_do_not_infer_edges() -> None:
    graph = BehaviorGraph(fragment())

    paths = graph.paths("entity.source", "entity.sink")
    derivations = graph.derivation_paths("entity.source", "entity.cycle")
    sequences = graph.operation_sequences("operation.read", "operation.archive")
    no_inferred_sequence = graph.operation_sequences("operation.read", "operation.unrelated")

    assert [path.entity_ids for path in paths.paths] == [
        ("entity.source", "entity.branch", "entity.sink"),
        ("entity.source", "entity.derived", "entity.sink"),
    ]
    assert [path.entity_ids for path in derivations.paths] == [
        ("entity.source", "entity.derived", "entity.cycle"),
    ]
    assert [sequence.operation_ids for sequence in sequences.sequences] == [
        ("operation.read", "operation.send", "operation.archive"),
    ]
    assert no_inferred_sequence.sequences == ()


def test_graph_order_is_canonical_across_repeated_construction() -> None:
    original = fragment()
    reverse = original.model_copy(
        update={
            "entities": tuple(reversed(original.entities)),
            "operations": tuple(reversed(original.operations)),
            "relationships": tuple(reversed(original.relationships)),
        }
    )

    first = BehaviorGraph(original)
    second = BehaviorGraph(reverse)

    assert first.direct_relationships() == second.direct_relationships()
    assert first.paths("entity.source", "entity.sink") == second.paths(
        "entity.source", "entity.sink"
    )
    assert first.operation_sequences(
        "operation.read", "operation.archive"
    ) == second.operation_sequences("operation.read", "operation.archive")


def test_max_depth_bounds_path_search_when_a_caller_supplies_one() -> None:
    graph = BehaviorGraph(fragment())

    unbounded = graph.paths("entity.source", "entity.sink")
    depth_limited = graph.paths("entity.source", "entity.sink", max_depth=1)

    assert len(unbounded.paths) == 2
    assert depth_limited.paths == ()


def test_cycles_terminate_without_any_caller_supplied_depth() -> None:
    graph = BehaviorGraph(fragment())

    cycle_search = graph.derivation_paths("entity.source", "entity.cycle")

    assert cycle_search.paths == (cycle_search.paths[0],)


def test_constructor_rejects_unresolved_canonical_references() -> None:
    incomplete = fragment().model_copy(
        update={
            "relationships": (
                relationship(
                    "relationship.invalid",
                    "entity.source",
                    RelationType.DERIVED_FROM,
                    "entity.missing",
                ),
            )
        }
    )

    with pytest.raises(ValueError, match="unknown entity"):
        BehaviorGraph(incomplete)
