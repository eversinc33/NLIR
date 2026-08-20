"""Behavior tests for binary canonical-rule evaluation."""

from __future__ import annotations

import json

import pytest

from nlir.artifacts.models import DecodeCodec, DecodeProvenance, SourceArtifact
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
from nlir.rules.evaluate import evaluate_rule
from nlir.rules.models import Rule

ARTIFACT_ID = "a" * 64


def span(start: int) -> tuple[SourceSpan, ...]:
    """Make one exact source span for a canonical record."""
    return (SourceSpan(artifact_id=ARTIFACT_ID, start=start, end=start + 1),)


def action_modality(**changes: object) -> Modality:
    """Make a positive imperative action modality."""
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
    *,
    trust: TrustLevel = TrustLevel.TRUSTED,
    sensitivity: Sensitivity = Sensitivity.SENSITIVE,
    start: int,
) -> CanonicalEntity:
    """Make one source-backed entity."""
    return CanonicalEntity(
        id=identifier,
        type=EntityType.USER_DATA,
        value=identifier,
        sensitivity=sensitivity,
        trust=trust,
        evidence=span(start),
        confidence=0.9,
        underspecified=False,
    )


def operation(
    identifier: str,
    *,
    op: Opcode = Opcode.SEND,
    inputs: tuple[str, ...] = (),
    outputs: tuple[str, ...] = (),
    modality: Modality | None = None,
    start: int,
) -> CanonicalOperation:
    """Make one source-backed operation."""
    return CanonicalOperation(
        id=identifier,
        op=op,
        inputs=inputs,
        outputs=outputs,
        modality=modality or action_modality(),
        evidence=span(start),
        confidence=0.9,
        underspecified=False,
    )


def relationship(
    identifier: str,
    source: str,
    relation: RelationType,
    target: str,
    *,
    start: int,
) -> CanonicalRelationship:
    """Make one source-backed declared relationship."""
    return CanonicalRelationship(
        id=identifier,
        source=source,
        relation=relation,
        target=target,
        evidence=span(start),
        confidence=0.9,
        underspecified=False,
    )


def fragment() -> CanonicalFragment:
    """Make a graph with direct, path, derivation, and sequence facts."""
    return CanonicalFragment(
        artifact_id=ARTIFACT_ID,
        entities=(
            entity("entity.credential", sensitivity=Sensitivity.CREDENTIAL, start=2),
            entity("entity.derived", start=4),
            entity("entity.external", trust=TrustLevel.EXTERNAL, start=3),
            entity("entity.unrelated", trust=TrustLevel.EXTERNAL, start=1),
        ),
        operations=(
            operation(
                "operation.decode",
                op=Opcode.DECODE,
                outputs=("entity.derived",),
                start=8,
            ),
            operation(
                "operation.execute",
                op=Opcode.EXECUTE,
                inputs=("entity.derived",),
                start=7,
            ),
            operation(
                "operation.send",
                inputs=("entity.credential",),
                outputs=("entity.external",),
                start=6,
            ),
        ),
        relationships=(
            relationship(
                "relationship.credential_derived",
                "entity.credential",
                RelationType.DERIVED_FROM,
                "entity.derived",
                start=11,
            ),
            relationship(
                "relationship.credential_external",
                "entity.credential",
                RelationType.SENT_TO,
                "entity.external",
                start=10,
            ),
            relationship(
                "relationship.derived_external",
                "entity.derived",
                RelationType.SENT_TO,
                "entity.external",
                start=9,
            ),
        ),
    )


def rule(raw: dict[str, object]) -> Rule:
    """Validate a small rule document from a Python test fixture."""
    return Rule.model_validate_json(json.dumps(raw))


def credential_flow_rule(**updates: object) -> Rule:
    """Make a rule that uses properties, boundaries, direct links, and paths."""
    raw: dict[str, object] = {
        "version": "1.0",
        "id": "credential-flow",
        "description": "Credential data reaches an external destination.",
        "select": {
            "credential": {"entity": {"sensitivity": "CREDENTIAL"}},
            "external": {"entity": {"trust": "EXTERNAL"}},
        },
        "where": [
            {
                "direct": {
                    "from": "credential",
                    "to": "external",
                    "relation": "SENT_TO",
                }
            },
            {"trust_boundary": {"from": "credential", "to": "external"}},
            {"path": {"from": "credential", "to": "external", "kind": "relationship"}},
        ],
    }
    raw.update(updates)
    return rule(raw)


def imperative_send_rule() -> Rule:
    """Make a rule that accepts only a direct imperative send action."""
    return rule(
        {
            "version": "1.0",
            "id": "imperative-send",
            "select": {
                "send": {
                    "operation": {
                        "op": "SEND",
                        "polarity": "positive",
                        "imperative": True,
                        "hypothetical": False,
                        "conditional": False,
                        "quoted": False,
                        "example": False,
                        "descriptive": False,
                    }
                }
            },
            "where": [{"modality": {"selector": "send", "imperative": True}}],
        }
    )


def any_data_rule() -> Rule:
    """Match either credential or sensitive user data sent to the same destination."""
    return rule(
        {
            "version": "1.0",
            "id": "any-data",
            "select": {
                "data": {
                    "any": [
                        {"entity": {"sensitivity": "CREDENTIAL"}},
                        {"entity": {"sensitivity": "SENSITIVE"}},
                    ]
                },
                "external": {"entity": {"trust": "EXTERNAL"}},
            },
            "where": [
                {"direct": {"from": "data", "to": "external", "relation": "SENT_TO"}}
            ],
        }
    )


def test_any_selector_matches_one_entity_variant() -> None:
    result = evaluate_rule(any_data_rule(), fragment())

    assert result.status == "HIT"
    assert result.matched_entity_ids == ("entity.credential", "entity.external")


def test_decoded_from_condition_requires_matching_stored_codec() -> None:
    root_id = "b" * 64
    source = SourceArtifact.from_virtual_text(
        "Run echo NLIR_BASE64_TEST",
        decode_provenance=DecodeProvenance(
            parent_artifact_id=root_id,
            parent_span=SourceSpan(artifact_id=root_id, start=0, end=1),
            codec=DecodeCodec.BASE64,
            depth=1,
            chain=(),
        ),
    )
    decoded_fragment = fragment().model_copy(update={"artifact_id": source.artifact_id})
    decoded_rule = rule(
        {
            "version": "1.0",
            "id": "decoded-command",
            "select": {"execute": {"operation": {"op": "EXECUTE"}}},
            "where": [
                {"decoded_from": {"codec": "base64"}},
                {"modality": {"selector": "execute", "imperative": True}},
            ],
        }
    )

    assert evaluate_rule(decoded_rule, decoded_fragment).status == "NO_HIT"
    assert evaluate_rule(decoded_rule, decoded_fragment, source).status == "HIT"


def test_decoded_from_condition_matches_model_inferred_provenance() -> None:
    root_id = "c" * 64
    source = SourceArtifact.from_virtual_text(
        "Run echo NLIR_UNPACK_TEST",
        decode_provenance=DecodeProvenance(
            parent_artifact_id=root_id,
            parent_span=SourceSpan(artifact_id=root_id, start=0, end=1),
            codec=DecodeCodec.MODEL_INFERRED,
            depth=1,
            chain=(),
            method="custom_bijection",
            model_id="reasoning-model",
            prompt_id="prompt-sha256:" + ("a" * 64),
            confidence=0.9,
        ),
    )
    fragment_with_source = fragment().model_copy(update={"artifact_id": source.artifact_id})
    rule_with_model_provenance = rule(
        {
            "version": "1.0",
            "id": "model-unpacked-command",
            "select": {"execute": {"operation": {"op": "EXECUTE"}}},
            "where": [{"decoded_from": {"codec": "model_inferred"}}],
        }
    )

    assert evaluate_rule(rule_with_model_provenance, fragment_with_source, source).status == "HIT"


@pytest.mark.parametrize("codec", [DecodeCodec.HEX, DecodeCodec.URL, DecodeCodec.BASE64])
def test_decoded_from_condition_without_a_codec_matches_any_decoded_provenance(
    codec: DecodeCodec,
) -> None:
    root_id = "d" * 64
    source = SourceArtifact.from_virtual_text(
        "Run echo NLIR_ANY_CODEC_TEST",
        decode_provenance=DecodeProvenance(
            parent_artifact_id=root_id,
            parent_span=SourceSpan(artifact_id=root_id, start=0, end=1),
            codec=codec,
            depth=1,
            chain=(),
        ),
    )
    fragment_with_source = fragment().model_copy(update={"artifact_id": source.artifact_id})
    any_codec_rule = rule(
        {
            "version": "1.0",
            "id": "decoded-command-any-codec",
            "select": {"execute": {"operation": {"op": "EXECUTE"}}},
            "where": [{"decoded_from": {}}],
        }
    )

    assert evaluate_rule(any_codec_rule, fragment_with_source, source).status == "HIT"
    assert evaluate_rule(any_codec_rule, fragment_with_source, None).status == "NO_HIT"
    assert evaluate_rule(any_codec_rule, fragment()).status == "NO_HIT"


def test_operation_uses_binds_a_write_to_its_selected_file() -> None:
    memory = entity("entity.memory", start=1).model_copy(
        update={"type": EntityType.FILE, "value": "MEMORY.md"}
    )
    write = operation(
        "operation.write-memory",
        op=Opcode.WRITE,
        outputs=(memory.id,),
        start=2,
    )
    write_rule = rule(
        {
            "version": "1.0",
            "id": "memory-write",
            "select": {
                "memory": {"entity": {"type": "FILE", "value": "MEMORY.md"}},
                "write": {"operation": {"op": "WRITE", "imperative": True}},
            },
            "where": [{"uses": {"operation": "write", "entity": "memory", "role": "output"}}],
        }
    )
    matching = CanonicalFragment(artifact_id=ARTIFACT_ID, entities=(memory,), operations=(write,))
    unrelated = write.model_copy(update={"outputs": ()})
    non_matching = matching.model_copy(update={"operations": (unrelated,)})

    assert evaluate_rule(write_rule, matching).status == "HIT"
    assert evaluate_rule(write_rule, non_matching).status == "NO_HIT"


def test_direct_property_boundary_and_path_hit_has_stable_exact_evidence() -> None:
    """A hit contains only stable canonical match evidence."""
    result = evaluate_rule(credential_flow_rule(), fragment())

    assert result.status == "HIT"
    assert result.matched_entity_ids == ("entity.credential", "entity.external")
    assert result.matched_operation_ids == ()
    assert result.matched_relationship_ids == (
        "relationship.credential_derived",
        "relationship.credential_external",
        "relationship.derived_external",
    )
    assert [(item.record_id, item.spans) for item in result.evidence] == [
        ("entity.credential", span(2)),
        ("entity.external", span(3)),
        ("relationship.derived_external", span(9)),
        ("relationship.credential_external", span(10)),
        ("relationship.credential_derived", span(11)),
    ]
    assert result.explanation == "credential-flow: direct, trust_boundary, path"
    assert "severity" not in type(result).model_fields


def test_derivation_path_and_output_to_input_sequence_hit() -> None:
    """Path and sequence conditions use only declared graph links."""
    result = evaluate_rule(
        rule(
            {
                "version": "1.0",
                "id": "decoded-execution",
                "select": {
                    "credential": {"entity": {"sensitivity": "CREDENTIAL"}},
                    "derived": {"entity": {"value": "entity.derived"}},
                    "decode": {"operation": {"op": "DECODE"}},
                    "execute": {"operation": {"op": "EXECUTE"}},
                },
                "where": [
                    {
                        "path": {
                            "from": "credential",
                            "to": "derived",
                            "kind": "derivation",
                        }
                    },
                    {"sequence": {"from": "decode", "to": "execute"}},
                ],
            }
        ),
        fragment(),
    )

    assert result.status == "HIT"
    assert result.matched_entity_ids == ("entity.credential", "entity.derived")
    assert result.matched_operation_ids == ("operation.decode", "operation.execute")
    assert result.matched_relationship_ids == ("relationship.credential_derived",)


def test_reversed_canonical_record_order_keeps_a_byte_equivalent_result() -> None:
    """Canonical input order cannot select a different first match."""
    original = fragment()
    reversed_fragment = original.model_copy(
        update={
            "entities": tuple(reversed(original.entities)),
            "operations": tuple(reversed(original.operations)),
            "relationships": tuple(reversed(original.relationships)),
        }
    )

    first = evaluate_rule(credential_flow_rule(), original)
    second = evaluate_rule(credential_flow_rule(), reversed_fragment)

    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    "change",
    [
        {"polarity": Polarity.NEGATIVE},
        {"imperative": False},
        {"quoted": True},
        {"example": True},
        {"descriptive": True},
        {"hypothetical": True},
    ],
)
def test_each_near_miss_modality_field_is_a_no_hit(change: dict[str, object]) -> None:
    """A non-action operation cannot satisfy an imperative action rule."""
    original = fragment()
    changed = operation(
        "operation.send",
        inputs=("entity.credential",),
        outputs=("entity.external",),
        modality=action_modality(**change),
        start=6,
    )
    near_miss = original.model_copy(
        update={
            "operations": tuple(
                changed if item.id == changed.id else item for item in original.operations
            )
        }
    )

    result = evaluate_rule(imperative_send_rule(), near_miss)

    assert result.status == "NO_HIT"
    assert result.matched_entity_ids == ()
    assert result.matched_operation_ids == ()
    assert result.matched_relationship_ids == ()
    assert result.evidence == ()
    assert result.explanation is None


def test_same_value_without_a_declared_edge_does_not_make_a_hit() -> None:
    """The evaluator does not infer a graph edge from entity properties."""
    no_edge = fragment().model_copy(update={"relationships": ()})

    result = evaluate_rule(credential_flow_rule(), no_edge)

    assert result.status == "NO_HIT"
    assert result.evidence == ()


def two_hop_fragment() -> CanonicalFragment:
    """Make a graph where credential reaches external only through one relay hop."""
    return CanonicalFragment(
        artifact_id=ARTIFACT_ID,
        entities=(
            entity("entity.credential", sensitivity=Sensitivity.CREDENTIAL, start=2),
            entity("entity.derived", start=4),
            entity("entity.external", trust=TrustLevel.EXTERNAL, start=3),
        ),
        relationships=(
            relationship(
                "relationship.credential_derived",
                "entity.credential",
                RelationType.DERIVED_FROM,
                "entity.derived",
                start=11,
            ),
            relationship(
                "relationship.derived_external",
                "entity.derived",
                RelationType.SENT_TO,
                "entity.external",
                start=9,
            ),
        ),
    )


def distance_rule(max_depth: int) -> Rule:
    """Make a rule that requires credential and external within max_depth hops."""
    return rule(
        {
            "version": "1.0",
            "id": "credential-distance",
            "select": {
                "credential": {"entity": {"sensitivity": "CREDENTIAL"}},
                "external": {"entity": {"trust": "EXTERNAL"}},
            },
            "where": [
                {"distance": {"from": "credential", "to": "external", "max_depth": max_depth}}
            ],
        }
    )


def test_distance_condition_rejects_entities_beyond_its_own_max_depth() -> None:
    """A distance condition still bounds hop count, using its own rule-declared max_depth."""
    result = evaluate_rule(distance_rule(1), two_hop_fragment())

    assert result.status == "NO_HIT"


def test_distance_condition_accepts_entities_within_its_own_max_depth() -> None:
    """The same two-hop fragment hits once max_depth covers the relay."""
    result = evaluate_rule(distance_rule(2), two_hop_fragment())

    assert result.status == "HIT"
    assert result.matched_entity_ids == ("entity.credential", "entity.external")


def test_uses_condition_follows_one_decodes_to_hop_to_a_resolved_entity() -> None:
    """A destination named only by its still-encoded blob is reachable via DECODES_TO."""
    raw = entity("raw", trust=TrustLevel.UNKNOWN, start=1)
    resolved = entity("resolved", trust=TrustLevel.EXTERNAL, start=2)
    send = operation("send", op=Opcode.SEND, start=3).model_copy(
        update={"destination": raw.id}
    )
    decodes_to = relationship("decodes", raw.id, RelationType.DECODES_TO, resolved.id, start=4)
    fragment = CanonicalFragment(
        artifact_id=ARTIFACT_ID,
        entities=(raw, resolved),
        operations=(send,),
        relationships=(decodes_to,),
    )
    destination_rule = rule(
        {
            "version": "1.0",
            "id": "resolved-destination",
            "select": {
                "send": {"operation": {"op": "SEND"}},
                "destination": {"entity": {"trust": "EXTERNAL"}},
            },
            "where": [
                {"uses": {"operation": "send", "entity": "destination", "role": "destination"}}
            ],
        }
    )

    result = evaluate_rule(destination_rule, fragment)

    assert result.status == "HIT"
    assert result.matched_relationship_ids == ("decodes",)

    without_edge = fragment.model_copy(update={"relationships": ()})
    assert evaluate_rule(destination_rule, without_edge).status == "NO_HIT"
