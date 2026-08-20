"""Regression tests for the closed, evidence-backed security IR."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nlir.artifacts.models import SourceArtifact
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


def evidence() -> tuple[SourceSpan, ...]:
    artifact = SourceArtifact.from_text("Read DEMO_TOKEN", source_name="fixture.txt")
    return (SourceSpan(artifact_id=artifact.artifact_id, start=0, end=4),)


def modality() -> Modality:
    return Modality(
        polarity=Polarity.POSITIVE,
        imperative=True,
        hypothetical=False,
        conditional=False,
        quoted=False,
        example=False,
        descriptive=False,
    )


def entity(**overrides: object) -> Entity:
    values: dict[str, object] = {
        "id": "credential-1",
        "type": EntityType.CREDENTIAL,
        "subtype": None,
        "value": "DEMO_TOKEN",
        "sensitivity": Sensitivity.CREDENTIAL,
        "trust": TrustLevel.TRUSTED,
        "evidence": evidence(),
        "confidence": 0.8,
        "underspecified": False,
    }
    values.update(overrides)
    return Entity(**values)


def operation(**overrides: object) -> Operation:
    values: dict[str, object] = {
        "id": "read-1",
        "op": Opcode.READ,
        "actor": None,
        "inputs": ("credential-1",),
        "outputs": (),
        "destination": None,
        "modality": modality(),
        "evidence": evidence(),
        "confidence": 0.8,
        "underspecified": False,
    }
    values.update(overrides)
    return Operation(**values)


def relationship(**overrides: object) -> Relationship:
    values: dict[str, object] = {
        "source": "credential-1",
        "relation": RelationType.CONSUMES,
        "target": "read-1",
        "evidence": evidence(),
        "confidence": 0.8,
        "underspecified": False,
    }
    values.update(overrides)
    return Relationship(**values)


def test_empty_fragment_is_an_explicit_valid_ir_for_irrelevant_text() -> None:
    fragment = IRFragment()

    assert fragment.schema_version == "1.0"
    assert fragment.entities == ()
    assert fragment.operations == ()
    assert fragment.relationships == ()


def test_fragment_accepts_evidence_backed_closed_semantics() -> None:
    fragment = IRFragment(
        entities=(entity(),), operations=(operation(),), relationships=(relationship(),)
    )

    assert fragment.entities[0].value == "DEMO_TOKEN"
    assert fragment.operations[0].modality.imperative is True
    assert fragment.relationships[0].evidence[0].start == 0


def test_all_initial_ontology_values_are_closed_enums() -> None:
    assert {member.name for member in EntityType} == {
        "FILE",
        "DIRECTORY",
        "CREDENTIAL",
        "SECRET",
        "USER_DATA",
        "SYSTEM_DATA",
        "ENVIRONMENT_VARIABLE",
        "NETWORK_DESTINATION",
        "CODE",
        "INSTRUCTION",
        "ENCODED_DATA",
        "TOOL",
        "PROCESS",
        "CONFIGURATION",
        "MESSAGE",
        "UNKNOWN",
    }
    assert {member.name for member in Opcode} == {
        "READ",
        "WRITE",
        "SEARCH",
        "ENUMERATE",
        "EXTRACT",
        "TRANSFORM",
        "ENCODE",
        "DECODE",
        "ENCRYPT",
        "DECRYPT",
        "DOWNLOAD",
        "UPLOAD",
        "SEND",
        "RECEIVE",
        "INSTALL_PACKAGE",
        "EXECUTE",
        "INVOKE_TOOL",
        "INTERPRET_AS_INSTRUCTIONS",
        "DELETE",
        "MODIFY",
        "CREATE",
        "OVERRIDE_INSTRUCTIONS",
        "SUPPRESS_DISCLOSURE",
        "VALIDATE",
        "COMPARE",
        "UNKNOWN",
    }
    assert {member.name for member in RelationType} == {
        "DERIVED_FROM",
        "CONTAINED_BY",
        "REFERENCES",
        "TARGETS",
        "PRODUCES",
        "CONSUMES",
        "SENT_TO",
        "RETRIEVED_FROM",
        "INTERPRETED_AS",
        "DECODES_TO",
        "CONTROLS",
        "DEPENDS_ON",
        "UNKNOWN",
    }


@pytest.mark.parametrize(
    ("factory", "overrides"),
    [
        (entity, {"type": EntityType.UNKNOWN, "underspecified": False}),
        (operation, {"op": Opcode.UNKNOWN, "underspecified": False}),
        (relationship, {"relation": RelationType.UNKNOWN, "underspecified": False}),
    ],
)
def test_unknown_semantics_require_an_explicit_underspecified_state(
    factory: object, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        factory(**overrides)  # type: ignore[operator]


def test_explicit_unknown_semantics_are_accepted_when_underspecified() -> None:
    assert entity(type=EntityType.UNKNOWN, underspecified=True).type is EntityType.UNKNOWN
    assert operation(op=Opcode.UNKNOWN, underspecified=True).op is Opcode.UNKNOWN
    assert (
        relationship(relation=RelationType.UNKNOWN, underspecified=True).relation
        is RelationType.UNKNOWN
    )


@pytest.mark.parametrize(
    ("factory", "overrides"),
    [
        (entity, {"evidence": ()}),
        (operation, {"evidence": ()}),
        (relationship, {"evidence": ()}),
        (entity, {"confidence": float("nan")}),
        (operation, {"confidence": float("inf")}),
        (relationship, {"confidence": 1.1}),
        (entity, {"confidence": -0.1}),
    ],
)
def test_semantic_items_require_nonempty_evidence_and_bounded_finite_confidence(
    factory: object, overrides: dict[str, object]
) -> None:
    with pytest.raises(ValidationError):
        factory(**overrides)  # type: ignore[operator]


def test_operation_requires_every_modality_field_and_evidence() -> None:
    incomplete_modality = {
        "polarity": "positive",
        "imperative": True,
        "hypothetical": False,
        "conditional": False,
        "quoted": False,
        "example": False,
        "descriptive": False,
    }

    with pytest.raises(ValidationError):
        operation(modality=Modality.model_validate(incomplete_modality))
    with pytest.raises(ValidationError):
        Operation.model_validate({**operation().model_dump(), "evidence": ()})


@pytest.mark.parametrize(
    "invalid",
    [
        {"type": "credential"},
        {"type": "not_an_ontology_value"},
        {"confidence": "0.8"},
        {"evidence": []},
        {"extra": "not allowed"},
    ],
)
def test_ir_rejects_type_coercion_unsupported_enums_and_extra_fields(
    invalid: dict[str, object],
) -> None:
    payload = entity().model_dump()
    payload.update(invalid)

    with pytest.raises(ValidationError):
        Entity.model_validate(payload)


def test_value_cannot_be_a_multiline_opaque_source_blob() -> None:
    with pytest.raises(ValidationError):
        entity(value="This is an unparsed source sentence\\nwith another line.")
