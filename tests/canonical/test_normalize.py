"""Contract tests for deterministic, evidence-preserving canonicalization."""

from __future__ import annotations

import pytest

from nlir.artifacts.models import SourceArtifact
from nlir.canonical.normalize import normalize_fragment
from nlir.contracts.common import SourceSpan
from nlir.contracts.ir import (
    Entity,
    EntityType,
    IRFragment,
    Modality,
    Opcode,
    Operation,
    Polarity,
    Relationship,
    RelationType,
    Sensitivity,
    TrustLevel,
)


def source() -> SourceArtifact:
    return SourceArtifact.from_text(
        "Read DEMO_TOKEN then send it to example.invalid.", source_name="case.md"
    )


def other_source() -> SourceArtifact:
    return SourceArtifact.from_text("A separate source.", source_name="other.md")


def span(artifact: SourceArtifact, start: int, end: int) -> SourceSpan:
    return SourceSpan(artifact_id=artifact.artifact_id, start=start, end=end)


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


def credential(
    artifact: SourceArtifact, *, identifier: str = "credential-1", **changes: object
) -> Entity:
    values: dict[str, object] = {
        "id": identifier,
        "type": EntityType.CREDENTIAL,
        "subtype": None,
        "value": "DEMO_TOKEN",
        "sensitivity": Sensitivity.CREDENTIAL,
        "trust": TrustLevel.TRUSTED,
        "evidence": (span(artifact, 5, 15),),
        "confidence": 0.9,
        "underspecified": False,
    }
    values.update(changes)
    return Entity(**values)


def destination(artifact: SourceArtifact, *, identifier: str = "destination-1") -> Entity:
    return Entity(
        id=identifier,
        type=EntityType.NETWORK_DESTINATION,
        subtype=None,
        value="example.invalid",
        sensitivity=Sensitivity.NONE,
        trust=TrustLevel.EXTERNAL,
        evidence=(span(artifact, 32, 47),),
        confidence=0.9,
        underspecified=False,
    )


def operation(
    artifact: SourceArtifact, *, identifier: str = "send-1", **changes: object
) -> Operation:
    values: dict[str, object] = {
        "id": identifier,
        "op": Opcode.SEND,
        "actor": None,
        "inputs": ("credential-1",),
        "outputs": (),
        "destination": "destination-1",
        "modality": modality(),
        "evidence": (span(artifact, 21, 25),),
        "confidence": 0.9,
        "underspecified": False,
    }
    values.update(changes)
    return Operation(**values)


def relationship(artifact: SourceArtifact, **changes: object) -> Relationship:
    values: dict[str, object] = {
        "source": "credential-1",
        "relation": RelationType.SENT_TO,
        "target": "destination-1",
        "evidence": (span(artifact, 21, 47),),
        "confidence": 0.9,
        "underspecified": False,
    }
    values.update(changes)
    return Relationship(**values)


def registry(*artifacts: SourceArtifact) -> dict[str, SourceArtifact]:
    return {artifact.artifact_id: artifact for artifact in artifacts}


def test_reconciles_exact_same_artifact_facts_with_readable_stable_ids_and_evidence() -> None:
    artifact = source()
    fragment = IRFragment(
        entities=(
            credential(artifact),
            credential(artifact, identifier="renamed-credential", evidence=(span(artifact, 0, 4),)),
            destination(artifact),
        ),
        operations=(
            operation(artifact),
            operation(
                artifact,
                identifier="send-2",
                inputs=("renamed-credential",),
                evidence=(span(artifact, 26, 30),),
            ),
        ),
        relationships=(
            relationship(artifact),
            relationship(artifact, source="renamed-credential", evidence=(span(artifact, 0, 47),)),
        ),
    )

    first = normalize_fragment(fragment, artifact.artifact_id, registry(artifact))
    renamed = normalize_fragment(
        fragment.model_copy(
            update={
                "entities": (
                    fragment.entities[0].model_copy(update={"id": "credential-renamed-again"}),
                    *fragment.entities[1:],
                ),
                "operations": (
                    fragment.operations[0].model_copy(
                        update={"inputs": ("credential-renamed-again",)}
                    ),
                    *fragment.operations[1:],
                ),
                "relationships": (
                    fragment.relationships[0].model_copy(
                        update={"source": "credential-renamed-again"}
                    ),
                    *fragment.relationships[1:],
                ),
            }
        ),
        artifact.artifact_id,
        registry(artifact),
    )

    assert first.diagnostics == ()
    assert first.fragment is not None
    assert first.fragment == renamed.fragment
    assert {entry.canonical_id for entry in first.source_to_canonical} == {
        entry.canonical_id for entry in renamed.source_to_canonical
    }
    assert len(first.fragment.entities) == 2
    assert len(first.fragment.operations) == 1
    assert len(first.fragment.relationships) == 1
    credential_entity = next(
        entity for entity in first.fragment.entities if entity.value == "DEMO_TOKEN"
    )
    assert credential_entity.id.startswith("entity.credential.demo_token.")
    assert len(credential_entity.id.rsplit(".", 1)[1]) == 12
    assert credential_entity.evidence == (span(artifact, 0, 4), span(artifact, 5, 15))
    canonical_operation = first.fragment.operations[0]
    canonical_relationship = first.fragment.relationships[0]
    assert canonical_operation.inputs == (credential_entity.id,)
    assert canonical_relationship.source == credential_entity.id
    assert canonical_operation.id.startswith("operation.send.")
    assert canonical_relationship.id.startswith("relationship.sent_to.")


@pytest.mark.parametrize(
    "conflicting_entity",
    [
        {"subtype": "api_token"},
        {"sensitivity": Sensitivity.SECRET},
        {"trust": TrustLevel.UNTRUSTED},
        {"confidence": 0.8},
        {"underspecified": True},
    ],
)
def test_rejects_an_entire_candidate_for_incompatible_same_key_entity_properties(
    conflicting_entity: dict[str, object],
) -> None:
    artifact = source()
    fragment = IRFragment(
        entities=(
            credential(artifact),
            credential(artifact, identifier="credential-2", **conflicting_entity),
        )
    )

    result = normalize_fragment(fragment, artifact.artifact_id, registry(artifact))

    assert result.fragment is None
    assert result.source_to_canonical == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "entity_reconciliation_conflict"
    ]


def test_rejects_conflicting_operation_modality_and_relationship_properties_atomically() -> None:
    artifact = source()
    entities = (credential(artifact), destination(artifact))
    operation_conflict = IRFragment(
        entities=entities,
        operations=(
            operation(artifact),
            operation(artifact, identifier="send-2", modality=modality(quoted=True)),
        ),
    )
    relationship_conflict = IRFragment(
        entities=entities,
        relationships=(relationship(artifact), relationship(artifact, confidence=0.8)),
    )

    operation_result = normalize_fragment(
        operation_conflict, artifact.artifact_id, registry(artifact)
    )
    relationship_result = normalize_fragment(
        relationship_conflict, artifact.artifact_id, registry(artifact)
    )

    assert operation_result.fragment is None
    assert [diagnostic.code for diagnostic in operation_result.diagnostics] == [
        "operation_reconciliation_conflict"
    ]
    assert relationship_result.fragment is None
    assert [diagnostic.code for diagnostic in relationship_result.diagnostics] == [
        "relationship_reconciliation_conflict"
    ]


def test_keeps_value_less_entities_distinct_and_rejects_cross_artifact_evidence() -> None:
    artifact = source()
    second = other_source()
    valueless = Entity(
        id="unknown-1",
        type=EntityType.UNKNOWN,
        subtype=None,
        value=None,
        sensitivity=Sensitivity.UNKNOWN,
        trust=TrustLevel.UNKNOWN,
        evidence=(span(artifact, 0, 4),),
        confidence=0.2,
        underspecified=True,
    )
    fragment = IRFragment(entities=(valueless, valueless.model_copy(update={"id": "unknown-2"})))

    accepted = normalize_fragment(fragment, artifact.artifact_id, registry(artifact))
    cross_artifact = normalize_fragment(
        IRFragment(
            entities=(credential(artifact).model_copy(update={"evidence": (span(second, 0, 1),)}),)
        ),
        artifact.artifact_id,
        registry(artifact, second),
    )

    assert accepted.fragment is not None
    assert len(accepted.fragment.entities) == 2
    assert accepted.fragment.operations == ()
    assert accepted.fragment.relationships == ()
    assert cross_artifact.fragment is None
    assert [diagnostic.code for diagnostic in cross_artifact.diagnostics] == [
        "cross_artifact_evidence"
    ]
