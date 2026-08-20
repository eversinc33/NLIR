"""Closed, versioned security semantics with mandatory source evidence."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat, model_validator

from nlir.contracts.common import SourceSpan, StrictFrozenModel


class EntityType(StrEnum):
    FILE = "FILE"
    DIRECTORY = "DIRECTORY"
    CREDENTIAL = "CREDENTIAL"
    SECRET = "SECRET"
    USER_DATA = "USER_DATA"
    SYSTEM_DATA = "SYSTEM_DATA"
    ENVIRONMENT_VARIABLE = "ENVIRONMENT_VARIABLE"
    NETWORK_DESTINATION = "NETWORK_DESTINATION"
    CODE = "CODE"
    INSTRUCTION = "INSTRUCTION"
    ENCODED_DATA = "ENCODED_DATA"
    TOOL = "TOOL"
    PROCESS = "PROCESS"
    CONFIGURATION = "CONFIGURATION"
    MESSAGE = "MESSAGE"
    UNKNOWN = "UNKNOWN"


class Sensitivity(StrEnum):
    NONE = "NONE"
    INTERNAL = "INTERNAL"
    SENSITIVE = "SENSITIVE"
    SECRET = "SECRET"
    CREDENTIAL = "CREDENTIAL"
    UNKNOWN = "UNKNOWN"


class TrustLevel(StrEnum):
    TRUSTED = "TRUSTED"
    UNTRUSTED = "UNTRUSTED"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


class Opcode(StrEnum):
    READ = "READ"
    WRITE = "WRITE"
    SEARCH = "SEARCH"
    ENUMERATE = "ENUMERATE"
    EXTRACT = "EXTRACT"
    TRANSFORM = "TRANSFORM"
    ENCODE = "ENCODE"
    DECODE = "DECODE"
    ENCRYPT = "ENCRYPT"
    DECRYPT = "DECRYPT"
    DOWNLOAD = "DOWNLOAD"
    UPLOAD = "UPLOAD"
    SEND = "SEND"
    RECEIVE = "RECEIVE"
    INSTALL_PACKAGE = "INSTALL_PACKAGE"
    EXECUTE = "EXECUTE"
    INVOKE_TOOL = "INVOKE_TOOL"
    INTERPRET_AS_INSTRUCTIONS = "INTERPRET_AS_INSTRUCTIONS"
    DELETE = "DELETE"
    MODIFY = "MODIFY"
    CREATE = "CREATE"
    OVERRIDE_INSTRUCTIONS = "OVERRIDE_INSTRUCTIONS"
    SUPPRESS_DISCLOSURE = "SUPPRESS_DISCLOSURE"
    VALIDATE = "VALIDATE"
    COMPARE = "COMPARE"
    UNKNOWN = "UNKNOWN"


class RelationType(StrEnum):
    DERIVED_FROM = "DERIVED_FROM"
    CONTAINED_BY = "CONTAINED_BY"
    REFERENCES = "REFERENCES"
    TARGETS = "TARGETS"
    PRODUCES = "PRODUCES"
    CONSUMES = "CONSUMES"
    SENT_TO = "SENT_TO"
    RETRIEVED_FROM = "RETRIEVED_FROM"
    INTERPRETED_AS = "INTERPRETED_AS"
    DECODES_TO = "DECODES_TO"
    CONTROLS = "CONTROLS"
    DEPENDS_ON = "DEPENDS_ON"
    UNKNOWN = "UNKNOWN"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


SemanticId = Annotated[
    str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9_.:-]*$")
]
Confidence = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]
Evidence = Annotated[tuple[SourceSpan, ...], Field(min_length=1)]
NormalizedValue = Annotated[str, Field(min_length=1, max_length=256, pattern=r"^\S+$")]


class Modality(StrictFrozenModel):
    """How a source expresses an operation, including critical near-misses."""

    polarity: Polarity
    imperative: bool
    hypothetical: bool
    conditional: bool
    quoted: bool
    example: bool
    descriptive: bool


class Entity(StrictFrozenModel):
    """One evidence-backed participant in a security-relevant behavior."""

    id: SemanticId
    type: EntityType
    subtype: Annotated[str, Field(min_length=1, max_length=64)] | None = None
    value: NormalizedValue | None = None
    sensitivity: Sensitivity
    trust: TrustLevel
    evidence: Evidence
    confidence: Confidence
    underspecified: bool

    @model_validator(mode="after")
    def unknown_requires_underspecified(self) -> Entity:
        if self.type is EntityType.UNKNOWN and not self.underspecified:
            raise ValueError("UNKNOWN entities must be explicitly underspecified")
        return self


class Operation(StrictFrozenModel):
    """One evidence-backed action with complete modality context."""

    id: SemanticId
    op: Opcode
    actor: SemanticId | None = None
    inputs: tuple[SemanticId, ...] = ()
    outputs: tuple[SemanticId, ...] = ()
    destination: SemanticId | None = None
    modality: Modality
    evidence: Evidence
    confidence: Confidence
    underspecified: bool

    @model_validator(mode="after")
    def unknown_requires_underspecified(self) -> Operation:
        if self.op is Opcode.UNKNOWN and not self.underspecified:
            raise ValueError("UNKNOWN operations must be explicitly underspecified")
        return self


class Relationship(StrictFrozenModel):
    """An evidence-backed typed link between semantic identifiers."""

    source: SemanticId
    relation: RelationType
    target: SemanticId
    evidence: Evidence
    confidence: Confidence
    underspecified: bool

    @model_validator(mode="after")
    def unknown_requires_underspecified(self) -> Relationship:
        if self.relation is RelationType.UNKNOWN and not self.underspecified:
            raise ValueError("UNKNOWN relationships must be explicitly underspecified")
        return self


class IRFragment(StrictFrozenModel):
    """The initial frozen schema version; an empty fragment means irrelevant text."""

    schema_version: Literal["1.0"] = "1.0"
    entities: tuple[Entity, ...] = ()
    operations: tuple[Operation, ...] = ()
    relationships: tuple[Relationship, ...] = ()
