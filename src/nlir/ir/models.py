"""Strict contracts for the lifted IR of one source artifact."""

from __future__ import annotations

from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from nlir.artifacts.models import SourceArtifact
from nlir.contracts.common import ArtifactId, StrictFrozenModel
from nlir.contracts.diagnostics import Diagnostic
from nlir.lifting.models import CanonicalAttemptResult

IR_FORMAT = "1.0"


class LiftMetadata(StrictFrozenModel):
    """Required reproducibility data for one fixture lift."""

    ir_format: Literal["1.0"]
    canonical_schema_version: Literal["1.0"]
    normalizer_id: Literal["nlir.canonical.normalize:1.0"]
    extractor_id: Literal["nlir.artifacts.extract:1.0"]
    lifter_id: Literal["nlir.fixture_lifter:1.0"]
    model_id: Literal["none"]
    prompt_catalog_id: Annotated[str, Field(pattern=r"^fixture-catalog-sha256:[0-9a-f]{64}$")]


class LiveLiftMetadata(StrictFrozenModel):
    """Required safe reproducibility data for one live lift."""

    ir_format: Literal["1.0"]
    canonical_schema_version: Literal["1.0"]
    normalizer_id: Literal["nlir.canonical.normalize:1.0"]
    extractor_id: Literal["nlir.artifacts.extract:1.0"]
    lifter_id: Literal["nlir.live_responses_lifter:1.0"]
    model_id: Annotated[str, Field(min_length=1, max_length=200)]
    endpoint_id: Annotated[str, Field(min_length=8, max_length=2_048)]
    prompt_id: Annotated[str, Field(pattern=r"^prompt-sha256:[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def identifiers_must_be_safe_and_normalized(self) -> LiveLiftMetadata:
        if self.model_id != self.model_id.strip() or any(
            character in self.model_id for character in "\r\n\t"
        ):
            raise ValueError("live model identity is invalid")
        if _normalized_live_endpoint(self.endpoint_id) != self.endpoint_id:
            raise ValueError("live endpoint identity is invalid")
        return self


LiftRecordMetadata = LiftMetadata | LiveLiftMetadata


class LiftDiagnostic(StrictFrozenModel):
    """An IR-level diagnostic. It has no severity or finding value."""

    code: Annotated[str, Field(min_length=1, max_length=80, pattern=r"^[a-z][a-z0-9_]*$")]
    message: Annotated[str, Field(min_length=1, max_length=512)]
    artifact_id: ArtifactId | None = None


class SourceLocationHint(StrictFrozenModel):
    """One exact source location for evidence from a binary hit."""

    artifact_id: ArtifactId
    source_name: Annotated[str, Field(min_length=1, max_length=1024)]
    start: Annotated[int, Field(ge=0)]
    end: Annotated[int, Field(ge=1)]
    line: Annotated[int, Field(ge=1)]
    column: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def end_must_follow_start(self) -> SourceLocationHint:
        if self.end <= self.start:
            raise ValueError("hint end must follow its start")
        return self


class HuntResult(StrictFrozenModel):
    """One binary rule result for one accepted lift attempt."""

    artifact_id: ArtifactId
    attempt_ordinal: Annotated[int, Field(ge=0)]
    status: Literal["HIT", "NO_HIT"]
    hints: tuple[SourceLocationHint, ...] = ()

    @model_validator(mode="after")
    def only_hits_may_have_hints(self) -> HuntResult:
        if self.status == "NO_HIT" and self.hints:
            raise ValueError("a no-hit cannot have source hints")
        return self


class HuntReport(StrictFrozenModel):
    """All binary results and non-finding diagnostics for one rule run."""

    results: tuple[HuntResult, ...]
    diagnostics: tuple[LiftDiagnostic, ...] = ()


class ArtifactRecord(StrictFrozenModel):
    """The complete lifted IR for one source artifact."""

    source: SourceArtifact
    canonical_attempts: tuple[CanonicalAttemptResult, ...] = Field(min_length=1)
    scan_diagnostics: tuple[Diagnostic, ...] = ()
    decode_diagnostics: tuple[Diagnostic, ...] = ()
    metadata: LiftRecordMetadata

    @model_validator(mode="after")
    def validate_complete_record(self) -> ArtifactRecord:
        ordinals = [attempt.ordinal for attempt in self.canonical_attempts]
        if ordinals != sorted(ordinals) or len(ordinals) != len(set(ordinals)):
            raise ValueError("canonical attempts must have unique ascending ordinals")
        for attempt in self.canonical_attempts:
            fragment = attempt.canonical_fragment
            if fragment is not None:
                if fragment.artifact_id != self.source.artifact_id:
                    raise ValueError("canonical fragment must belong to its source artifact")
                if fragment.schema_version != self.metadata.canonical_schema_version:
                    raise ValueError("canonical fragment schema version is incompatible")
        return self


class LiftedIR(StrictFrozenModel):
    """One lift result, with every lifted root and decoded child record.

    The caller owns this value: serialize it with ``model_dump_json`` and read it
    back with ``model_validate_json``.  NLIR never persists it.
    """

    records: tuple[ArtifactRecord, ...]

    @property
    def artifact_ids(self) -> tuple[str, ...]:
        """Return the stable source identities from this lift."""
        return tuple(record.source.artifact_id for record in self.records)


def _normalized_live_endpoint(value: str) -> str:
    """Return one safe API-root identity without query or credential data."""
    if not value or value != value.strip() or "\\" in value:
        raise ValueError("endpoint identity is invalid")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("endpoint identity is invalid")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint identity is invalid")
    if parsed.query or parsed.fragment or parsed.path.startswith("//") or "//" in parsed.path:
        raise ValueError("endpoint identity is invalid")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("endpoint identity is invalid")
    path = parsed.path.rstrip("/")
    segments = [segment for segment in path.split("/") if segment]
    if any(segment in {".", ".."} or "%" in segment for segment in segments):
        raise ValueError("endpoint identity is invalid")
    if path.lower().endswith("/responses"):
        raise ValueError("endpoint identity is invalid")
    try:
        parsed.port
    except ValueError as error:
        raise ValueError("endpoint identity is invalid") from error
    return f"{parsed.scheme}://{parsed.netloc}{path}"
