"""Common strict models shared by semantic and non-semantic contracts."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictFrozenModel(BaseModel):
    """Base model for trusted NLIR contracts.

    Contracts are deliberately strict: callers must provide the intended type and
    no unknown data is retained or coerced into a fact.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


ArtifactId = Annotated[
    str,
    Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
        description="Lowercase SHA-256 content identifier.",
    ),
]


class SourceSpan(StrictFrozenModel):
    """A half-open, 0-based Python code-point range in one source artifact."""

    artifact_id: ArtifactId
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def end_must_follow_start(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("end must be greater than start for a non-empty source span")
        return self

    def extract(self, text: str) -> str:
        """Return this span's exact source text after checking it is in bounds."""
        if self.end > len(text):
            raise ValueError("source span exceeds source text length")
        return text[self.start : self.end]
