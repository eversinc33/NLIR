"""Tests for the atomic boundary around externally supplied IR fragments."""

from __future__ import annotations

from copy import deepcopy

import pytest

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
from nlir.contracts.validation import validate_fragment


def source() -> SourceArtifact:
    """Use CRLF and a non-ASCII code point to pin the offset convention."""
    return SourceArtifact.from_text(
        "Read DEMO_TOKEN\r\nthen send it to example.invalid\né", source_name="case.md"
    )


def span(artifact: SourceArtifact, start: int = 0, end: int = 4) -> SourceSpan:
    return SourceSpan(artifact_id=artifact.artifact_id, start=start, end=end)


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


def entity(
    artifact: SourceArtifact,
    *,
    identifier: str = "credential-1",
    entity_type: EntityType = EntityType.CREDENTIAL,
    value: str = "DEMO_TOKEN",
) -> Entity:
    return Entity(
        id=identifier,
        type=entity_type,
        subtype=None,
        value=value,
        sensitivity=Sensitivity.CREDENTIAL,
        trust=TrustLevel.TRUSTED,
        evidence=(span(artifact),),
        confidence=0.9,
        underspecified=False,
    )


def destination(artifact: SourceArtifact) -> Entity:
    return Entity(
        id="destination-1",
        type=EntityType.NETWORK_DESTINATION,
        subtype=None,
        value="example.invalid",
        sensitivity=Sensitivity.NONE,
        trust=TrustLevel.EXTERNAL,
        evidence=(span(artifact, 33, 48),),
        confidence=0.9,
        underspecified=False,
    )


def operation(
    artifact: SourceArtifact, *, identifier: str = "send-1", **overrides: object
) -> Operation:
    values: dict[str, object] = {
        "id": identifier,
        "op": Opcode.SEND,
        "actor": None,
        "inputs": ("credential-1",),
        "outputs": (),
        "destination": "destination-1",
        "modality": modality(),
        "evidence": (span(artifact, 22, 26),),
        "confidence": 0.9,
        "underspecified": False,
    }
    values.update(overrides)
    return Operation(**values)


def relationship(artifact: SourceArtifact, **overrides: object) -> Relationship:
    values: dict[str, object] = {
        "source": "credential-1",
        "relation": RelationType.SENT_TO,
        "target": "destination-1",
        "evidence": (span(artifact, 22, 48),),
        "confidence": 0.9,
        "underspecified": False,
    }
    values.update(overrides)
    return Relationship(**values)


def valid_fragment(artifact: SourceArtifact) -> IRFragment:
    return IRFragment(
        entities=(entity(artifact), destination(artifact)),
        operations=(operation(artifact),),
        relationships=(relationship(artifact),),
    )


def registry(artifact: SourceArtifact) -> dict[str, SourceArtifact]:
    return {artifact.artifact_id: artifact}


def test_accepts_a_complete_fragment_unchanged() -> None:
    artifact = source()
    fragment = valid_fragment(artifact)

    result = validate_fragment(fragment, registry(artifact))

    assert result.fragment is fragment
    assert result.diagnostics == ()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["entities"][0].update({"unexpected": True}),
        lambda raw: raw["entities"][0].update({"type": "NOT_AN_ENTITY"}),
        lambda raw: raw["operations"][0].pop("modality"),
    ],
)
def test_strict_shape_failures_are_atomic_and_diagnostic(mutate: object) -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    mutate(raw)  # type: ignore[operator]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_ir_shape"]


@pytest.mark.parametrize(
    "bad_span",
    [
        {"artifact_id": "f" * 64, "start": 0, "end": 1},
        {"artifact_id": "e" * 64, "start": 4, "end": 1},
        {"artifact_id": "d" * 64, "start": 0, "end": 999},
    ],
)
def test_missing_or_invalid_evidence_spans_are_rejected_atomically(
    bad_span: dict[str, object],
) -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    raw["entities"][0]["evidence"] = [bad_span]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_evidence_span"]


def test_evidence_uses_exact_python_code_point_offsets_including_crlf_and_unicode() -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    raw["entities"][0]["evidence"] = [{"artifact_id": artifact.artifact_id, "start": 49, "end": 50}]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is not None
    assert result.fragment.entities[0].evidence[0].extract(artifact.text) == "é"


@pytest.mark.parametrize(
    "duplicate",
    [
        lambda raw: raw["entities"].append(deepcopy(raw["entities"][0])),
        lambda raw: raw["operations"].append(deepcopy(raw["operations"][0])),
        lambda raw: raw["operations"][0].update({"id": "credential-1"}),
    ],
)
def test_duplicate_semantic_ids_are_rejected_atomically(duplicate: object) -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    duplicate(raw)  # type: ignore[operator]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["duplicate_semantic_id"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw["operations"][0].update({"actor": "missing-entity"}),
        lambda raw: raw["operations"][0].update({"inputs": ["missing-entity"]}),
        lambda raw: raw["operations"][0].update({"outputs": ["missing-entity"]}),
        lambda raw: raw["operations"][0].update({"destination": "missing-entity"}),
        lambda raw: raw["relationships"][0].update({"source": "missing-entity"}),
        lambda raw: raw["relationships"][0].update({"target": "missing-entity"}),
    ],
)
def test_dangling_semantic_references_are_rejected_atomically(mutate: object) -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    mutate(raw)  # type: ignore[operator]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["dangling_semantic_reference"]


def test_diagnostics_are_deterministically_ordered_when_multiple_invariants_fail() -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    raw["entities"].append(deepcopy(raw["entities"][0]))
    raw["operations"][0]["inputs"] = ["missing-entity"]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == [
        "duplicate_semantic_id",
        "dangling_semantic_reference",
    ]


def test_malformed_external_fragment_is_rejected_without_a_registry_span() -> None:
    result = validate_fragment({"unexpected": True}, {})

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_ir_shape"]
    assert result.diagnostics[0].span is None


def test_malformed_external_fragment_against_empty_source_has_no_fabricated_span() -> None:
    artifact = SourceArtifact.from_text("", source_name="empty.md")

    result = validate_fragment({"unexpected": True}, registry(artifact))

    assert result.fragment is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == ["invalid_ir_shape"]
    assert result.diagnostics[0].span is None


def test_unresolved_evidence_without_a_registry_is_rejected_without_a_span() -> None:
    artifact = source()

    result = validate_fragment(valid_fragment(artifact).model_dump(mode="json"), {})

    assert result.fragment is None
    assert result.diagnostics
    assert all(diagnostic.code == "invalid_evidence_span" for diagnostic in result.diagnostics)
    assert all(diagnostic.span is None for diagnostic in result.diagnostics)


def test_evidence_against_empty_source_is_rejected_without_a_fabricated_span() -> None:
    artifact = SourceArtifact.from_text("", source_name="empty.md")
    raw = valid_fragment(source()).model_dump(mode="json")
    for record in (*raw["entities"], *raw["operations"], *raw["relationships"]):
        for evidence in record["evidence"]:
            evidence["artifact_id"] = artifact.artifact_id

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert result.diagnostics
    assert all(diagnostic.code == "invalid_evidence_span" for diagnostic in result.diagnostics)
    assert all(diagnostic.span is None for diagnostic in result.diagnostics)


def test_present_rejection_span_is_in_bounds_for_non_empty_source() -> None:
    artifact = source()
    raw = valid_fragment(artifact).model_dump(mode="json")
    raw["entities"][0]["evidence"] = [{"artifact_id": artifact.artifact_id, "start": 0, "end": 999}]

    result = validate_fragment(raw, registry(artifact))

    assert result.fragment is None
    assert result.diagnostics
    assert all(diagnostic.span is not None for diagnostic in result.diagnostics)
    assert all(
        0 <= diagnostic.span.start < diagnostic.span.end <= len(artifact.text)
        for diagnostic in result.diagnostics
        if diagnostic.span is not None
    )


def test_accepts_empty_fragment_without_a_source_registry() -> None:
    fragment = IRFragment()

    result = validate_fragment(fragment, {})

    assert result.fragment is fragment
    assert result.diagnostics == ()
