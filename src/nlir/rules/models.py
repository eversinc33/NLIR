"""Strict, versioned contracts for small declarative hunting rules."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from nlir.artifacts.models import DecodeCodec
from nlir.contracts.common import SourceSpan, StrictFrozenModel
from nlir.contracts.ir import Opcode, RelationType
from nlir.graph.models import EntityFilter

RuleId = Annotated[
    str,
    Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9-]*$"),
]
RuleReference = Annotated[
    str,
    Field(min_length=12, max_length=2_048, pattern=r"^https://[^\s]+$"),
]
SelectorName = Annotated[
    str,
    Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$"),
]


class OperationPredicate(StrictFrozenModel):
    """Exact predicates for one declared operation and its modality."""

    op: Opcode | None = None
    polarity: Literal["positive", "negative", "unknown"] | None = None
    imperative: bool | None = None
    hypothetical: bool | None = None
    conditional: bool | None = None
    quoted: bool | None = None
    example: bool | None = None
    descriptive: bool | None = None

    @model_validator(mode="after")
    def must_declare_a_predicate(self) -> OperationPredicate:
        if not any(value is not None for value in self.model_dump().values()):
            raise ValueError("an operation selector must declare at least one predicate")
        return self


class EntitySelector(StrictFrozenModel):
    """A named selector for canonical entities."""

    entity: EntityFilter

    @model_validator(mode="after")
    def must_declare_a_predicate(self) -> EntitySelector:
        if not any(value is not None for value in self.entity.model_dump().values()):
            raise ValueError("an entity selector must declare at least one predicate")
        return self


class OperationSelector(StrictFrozenModel):
    """A named selector for canonical operations."""

    operation: OperationPredicate


class AnySelector(StrictFrozenModel):
    """Match any one of several entity or operation selector variants."""

    any: Annotated[tuple[EntitySelector | OperationSelector, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def alternatives_must_have_one_record_kind(self) -> AnySelector:
        if len({type(selector) for selector in self.any}) != 1:
            raise ValueError("an any selector cannot mix entity and operation variants")
        return self


Selector = EntitySelector | OperationSelector | AnySelector


class DirectConditionSpec(StrictFrozenModel):
    """Require one declared relationship between two entity selectors."""

    from_: SelectorName = Field(alias="from")
    to: SelectorName
    relation: RelationType


class DirectCondition(StrictFrozenModel):
    direct: DirectConditionSpec


class TrustBoundaryConditionSpec(StrictFrozenModel):
    """Require one declared relationship across a trust boundary."""

    from_: SelectorName = Field(alias="from")
    to: SelectorName


class TrustBoundaryCondition(StrictFrozenModel):
    trust_boundary: TrustBoundaryConditionSpec


class PathKind(StrEnum):
    RELATIONSHIP = "relationship"
    DERIVATION = "derivation"


class PathConditionSpec(StrictFrozenModel):
    """Require a bounded path between two entity selectors."""

    from_: SelectorName = Field(alias="from")
    to: SelectorName
    kind: PathKind = PathKind.RELATIONSHIP


class PathCondition(StrictFrozenModel):
    path: PathConditionSpec


class SequenceConditionSpec(StrictFrozenModel):
    """Require a declared output-to-input operation sequence."""

    from_: SelectorName = Field(alias="from")
    to: SelectorName


class SequenceCondition(StrictFrozenModel):
    sequence: SequenceConditionSpec


class ModalityConditionSpec(StrictFrozenModel):
    """Require exact modality values for an operation selector."""

    selector: SelectorName
    polarity: Literal["positive", "negative", "unknown"] | None = None
    imperative: bool | None = None
    hypothetical: bool | None = None
    conditional: bool | None = None
    quoted: bool | None = None
    example: bool | None = None
    descriptive: bool | None = None

    @model_validator(mode="after")
    def must_declare_a_predicate(self) -> ModalityConditionSpec:
        values = self.model_dump(exclude={"selector"})
        if not any(value is not None for value in values.values()):
            raise ValueError("a modality condition must declare at least one predicate")
        return self


class ModalityCondition(StrictFrozenModel):
    modality: ModalityConditionSpec


class OperationUsesConditionSpec(StrictFrozenModel):
    """Require one operation to reference one selected entity in a declared role."""

    operation: SelectorName
    entity: SelectorName
    role: Literal["actor", "input", "output", "destination", "any"] = "any"


class OperationUsesCondition(StrictFrozenModel):
    uses: OperationUsesConditionSpec


class DistanceConditionSpec(StrictFrozenModel):
    """Require two entities to be within a declared path depth."""

    from_: SelectorName = Field(alias="from")
    to: SelectorName
    max_depth: Annotated[int, Field(ge=1, le=4)]
    kind: PathKind = PathKind.RELATIONSHIP


class DistanceCondition(StrictFrozenModel):
    distance: DistanceConditionSpec


class DecodedFromConditionSpec(StrictFrozenModel):
    """Require the stored artifact to be a decoded child, of one exact codec or any codec."""

    codec: DecodeCodec | None = None


class DecodedFromCondition(StrictFrozenModel):
    decoded_from: DecodedFromConditionSpec


RuleCondition = (
    DirectCondition
    | TrustBoundaryCondition
    | PathCondition
    | SequenceCondition
    | ModalityCondition
    | OperationUsesCondition
    | DistanceCondition
    | DecodedFromCondition
)


class RuleMetadata(StrictFrozenModel):
    """Human-facing rule context that never affects binary evaluation."""

    description: Annotated[str, Field(min_length=1, max_length=512)]
    author: Annotated[str, Field(min_length=1, max_length=160)]
    references: tuple[RuleReference, ...] = ()

    @model_validator(mode="after")
    def text_fields_must_not_contain_control_characters(self) -> RuleMetadata:
        if any(character in self.description + self.author for character in "\r\n\t"):
            raise ValueError("rule metadata text is invalid")
        return self


class Rule(StrictFrozenModel):
    """A closed version-one rule with explicit selector alternatives."""

    version: Literal["1.0"]
    id: RuleId
    description: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    metadata: RuleMetadata | None = None
    select: Annotated[dict[SelectorName, Selector], Field(min_length=1)]
    where: Annotated[tuple[RuleCondition, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def selectors_must_exist_and_fit_conditions(self) -> Rule:
        for condition in self.where:
            if isinstance(
                condition,
                (DirectCondition, TrustBoundaryCondition, PathCondition, DistanceCondition),
            ):
                spec = next(iter(condition.model_dump(by_alias=True).values()))
                self._require_entity_selector(spec["from"])
                self._require_entity_selector(spec["to"])
            elif isinstance(condition, SequenceCondition):
                self._require_operation_selector(condition.sequence.from_)
                self._require_operation_selector(condition.sequence.to)
            elif isinstance(condition, DecodedFromCondition):
                continue
            elif isinstance(condition, OperationUsesCondition):
                self._require_operation_selector(condition.uses.operation)
                self._require_entity_selector(condition.uses.entity)
            else:
                self._require_operation_selector(condition.modality.selector)
        return self

    def _require_entity_selector(self, name: str) -> None:
        selector = self.select.get(name)
        if not isinstance(selector, EntitySelector) and not (
            isinstance(selector, AnySelector)
            and all(isinstance(variant, EntitySelector) for variant in selector.any)
        ):
            raise ValueError(f"condition references undeclared entity selector {name!r}")

    def _require_operation_selector(self, name: str) -> None:
        selector = self.select.get(name)
        if not isinstance(selector, OperationSelector) and not (
            isinstance(selector, AnySelector)
            and all(isinstance(variant, OperationSelector) for variant in selector.any)
        ):
            raise ValueError(f"condition references undeclared operation selector {name!r}")


class RuleDiagnostic(StrictFrozenModel):
    """A deterministic rule-load rejection that is not a rule result."""

    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    message: Annotated[str, Field(min_length=1, max_length=512)]
    span: SourceSpan | None = None


class RuleLoadResult(StrictFrozenModel):
    """Either one complete rule or deterministic error diagnostics."""

    rule: Rule | None
    diagnostics: tuple[RuleDiagnostic, ...] = ()

    @model_validator(mode="after")
    def must_be_accepted_or_rejected(self) -> RuleLoadResult:
        if self.rule is not None and self.diagnostics:
            raise ValueError("an accepted rule cannot include diagnostics")
        if self.rule is None and not self.diagnostics:
            raise ValueError("a rejected rule must include a diagnostic")
        return self


class MatchedRecord(StrictFrozenModel):
    """Exact source evidence from one canonical record in a rule match."""

    record_id: Annotated[str, Field(min_length=1, max_length=128)]
    record_type: Literal["entity", "operation", "relationship"]
    spans: tuple[SourceSpan, ...] = Field(min_length=1)


class RuleResult(StrictFrozenModel):
    """One binary result with evidence only when a complete match exists."""

    status: Literal["HIT", "NO_HIT"]
    matched_entity_ids: tuple[str, ...] = ()
    matched_operation_ids: tuple[str, ...] = ()
    matched_relationship_ids: tuple[str, ...] = ()
    evidence: tuple[MatchedRecord, ...] = ()
    explanation: str | None = None

    @model_validator(mode="after")
    def binary_results_keep_match_data_consistent(self) -> RuleResult:
        matched_ids = (
            self.matched_entity_ids,
            self.matched_operation_ids,
            self.matched_relationship_ids,
        )
        if self.status == "NO_HIT":
            if any(matched_ids) or self.evidence or self.explanation is not None:
                raise ValueError("a no-hit cannot contain matched evidence or an explanation")
        elif not any(matched_ids) or not self.evidence or self.explanation is None:
            raise ValueError("a hit requires matched IDs, evidence, and an explanation")
        return self
