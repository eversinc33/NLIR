"""Strict, immutable contracts for canonical graph inspection."""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import Field, model_validator

from nlir.canonical.models import (
    CanonicalEntity,
    CanonicalOperation,
    CanonicalRelationship,
)
from nlir.contracts.common import StrictFrozenModel
from nlir.contracts.ir import EntityType, Polarity, Sensitivity, TrustLevel


class EntityFilter(StrictFrozenModel):
    """Exact predicates over declared canonical entity fields.

    ``value_pattern``, unlike every other field here, is not an exact-equality
    predicate: it is a regular expression searched against ``value``.
    """

    type: EntityType | None = None
    subtype: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    value: Annotated[str, Field(min_length=1, max_length=256, pattern=r"^\S+$")] | None = None
    value_pattern: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    sensitivity: Sensitivity | None = None
    trust: TrustLevel | None = None
    underspecified: bool | None = None

    @model_validator(mode="after")
    def value_pattern_must_be_a_valid_regex(self) -> EntityFilter:
        if self.value_pattern is not None:
            try:
                re.compile(self.value_pattern)
            except re.error as error:
                raise ValueError(
                    f"value_pattern is not a valid regular expression: {error}"
                ) from error
        return self


class ModalityFilter(StrictFrozenModel):
    """Exact predicates over explicitly declared operation modality fields."""

    polarity: Polarity | None = None
    imperative: bool | None = None
    hypothetical: bool | None = None
    conditional: bool | None = None
    quoted: bool | None = None
    example: bool | None = None
    descriptive: bool | None = None


class RelationshipQueryResult(StrictFrozenModel):
    """Sorted declared relationship records matching a direct query."""

    records: tuple[CanonicalRelationship, ...] = ()


class EntityQueryResult(StrictFrozenModel):
    """Sorted declared entity records matching exact property predicates."""

    records: tuple[CanonicalEntity, ...] = ()


class OperationQueryResult(StrictFrozenModel):
    """Sorted declared operation records matching exact modality predicates."""

    records: tuple[CanonicalOperation, ...] = ()


class GraphPath(StrictFrozenModel):
    """One ordered path over declared relationships, retaining relationship evidence."""

    entity_ids: tuple[str, ...] = Field(min_length=1)
    relationships: tuple[CanonicalRelationship, ...] = ()

    @model_validator(mode="after")
    def relationship_count_matches_edges(self) -> GraphPath:
        if len(self.relationships) != len(self.entity_ids) - 1:
            raise ValueError("a graph path needs exactly one relationship for every edge")
        return self


class PathQueryResult(StrictFrozenModel):
    """Stably ordered declared relationship paths."""

    paths: tuple[GraphPath, ...] = ()


class OperationSequence(StrictFrozenModel):
    """Declared output-to-input operation sequence with source evidence intact."""

    operation_ids: tuple[str, ...] = Field(min_length=1)
    operations: tuple[CanonicalOperation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def operation_ids_match_records(self) -> OperationSequence:
        if self.operation_ids != tuple(operation.id for operation in self.operations):
            raise ValueError("operation IDs must match the returned operation records")
        return self


class SequenceQueryResult(StrictFrozenModel):
    """Stably ordered declared operation sequences."""

    sequences: tuple[OperationSequence, ...] = ()
