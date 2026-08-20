"""Contract tests for deterministic, serializable lifted IR."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nlir.artifacts.models import DecodeCodec, DecodeProvenance, SourceArtifact
from nlir.canonical.models import CanonicalFragment
from nlir.contracts.common import SourceSpan
from nlir.contracts.diagnostics import DiagnosticSeverity
from nlir.ir.models import (
    IR_FORMAT,
    ArtifactRecord,
    LiftedIR,
    LiftMetadata,
    LiveLiftMetadata,
)
from nlir.lifting.models import (
    CanonicalAttemptResult,
    CanonicalAttemptStage,
    LifterDiagnostic,
    LifterStage,
)


def metadata() -> LiftMetadata:
    return LiftMetadata(
        ir_format=IR_FORMAT,
        canonical_schema_version="1.0",
        normalizer_id="nlir.canonical.normalize:1.0",
        extractor_id="nlir.artifacts.extract:1.0",
        lifter_id="nlir.fixture_lifter:1.0",
        model_id="none",
        prompt_catalog_id="fixture-catalog-sha256:" + ("a" * 64),
    )


def live_metadata() -> LiveLiftMetadata:
    """Return safe reproducibility data for one fixed live lifter run."""
    return LiveLiftMetadata(
        ir_format=IR_FORMAT,
        canonical_schema_version="1.0",
        normalizer_id="nlir.canonical.normalize:1.0",
        extractor_id="nlir.artifacts.extract:1.0",
        lifter_id="nlir.live_responses_lifter:1.0",
        model_id="test-model",
        endpoint_id="https://api.example.invalid/v1",
        prompt_id="prompt-sha256:" + ("b" * 64),
    )


def attempts(artifact: SourceArtifact) -> tuple[CanonicalAttemptResult, ...]:
    fragment = CanonicalFragment(artifact_id=artifact.artifact_id)
    accepted = CanonicalAttemptResult(
        ordinal=0,
        stage=CanonicalAttemptStage.ACCEPTED,
        canonical_fragment=fragment,
    )
    rejected = CanonicalAttemptResult(
        ordinal=1,
        stage=CanonicalAttemptStage.LIFECYCLE_REJECTED,
        diagnostics=(
            LifterDiagnostic(
                stage=LifterStage.LIFECYCLE,
                code="fixture_refused",
                severity=DiagnosticSeverity.ERROR,
                message="The fixture refused the request.",
            ),
        ),
    )
    return (accepted, rejected)


def record(artifact: SourceArtifact | None = None) -> ArtifactRecord:
    source = artifact or SourceArtifact.from_text("lift this text", source_name="input.md")
    return ArtifactRecord(
        source=source,
        canonical_attempts=attempts(source),
        scan_diagnostics=(),
        decode_diagnostics=(),
        metadata=metadata(),
    )


def test_lifted_ir_round_trips_through_json_for_caller_owned_storage() -> None:
    lifted = LiftedIR(records=(record(),))

    restored = LiftedIR.model_validate_json(lifted.model_dump_json())

    assert restored == lifted
    assert restored.artifact_ids == lifted.artifact_ids
    assert restored.records[0].source.text == "lift this text"
    assert restored.model_dump_json() == lifted.model_dump_json()


def test_lifted_ir_round_trips_live_metadata_and_decoded_child_lineage() -> None:
    parent = SourceArtifact.from_text("YWJj", source_name="parent.md")
    provenance = DecodeProvenance(
        parent_artifact_id=parent.artifact_id,
        parent_span=SourceSpan(artifact_id=parent.artifact_id, start=0, end=4),
        codec=DecodeCodec.BASE64,
        depth=1,
        chain=(),
    )
    child = SourceArtifact.from_virtual_text("abc", decode_provenance=provenance)
    lifted = LiftedIR(
        records=(
            record(parent).model_copy(update={"metadata": live_metadata()}),
            record(child).model_copy(update={"metadata": live_metadata()}),
        )
    )

    restored = LiftedIR.model_validate_json(lifted.model_dump_json())

    assert restored == lifted
    assert restored.records[1].source.decode_provenance == provenance
    assert restored.records[1].metadata.model_dump() == {
        "ir_format": "1.0",
        "canonical_schema_version": "1.0",
        "normalizer_id": "nlir.canonical.normalize:1.0",
        "extractor_id": "nlir.artifacts.extract:1.0",
        "lifter_id": "nlir.live_responses_lifter:1.0",
        "model_id": "test-model",
        "endpoint_id": "https://api.example.invalid/v1",
        "prompt_id": "prompt-sha256:" + ("b" * 64),
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"endpoint_id": "https://api.example.invalid/v1?key=secret"},
        {"endpoint_id": "https://api-key@api.example.invalid/v1"},
        {"endpoint_id": "https://api.example.invalid/v1/responses"},
        {"prompt_id": "missing"},
        {"lifter_id": "nlir.fixture_lifter:1.0"},
        {"prompt_catalog_id": "fixture-catalog-sha256:" + ("a" * 64)},
        {"api_key": "credential-marker-not-for-output"},
        {"authorization": "Bearer credential-marker-not-for-output"},
    ],
)
def test_live_metadata_rejects_unsafe_and_fixture_only_values(changes: dict[str, str]) -> None:
    values = live_metadata().model_dump()
    values.update(changes)

    with pytest.raises(ValidationError):
        LiveLiftMetadata.model_validate(values)


def test_record_rejects_invalid_metadata_and_nonordered_attempts() -> None:
    source = SourceArtifact.from_text("lift this text", source_name="input.md")

    with pytest.raises(ValidationError):
        LiftMetadata(
            ir_format=IR_FORMAT,
            canonical_schema_version="1.0",
            normalizer_id="nlir.canonical.normalize:1.0",
            extractor_id="nlir.artifacts.extract:1.0",
            lifter_id="nlir.fixture_lifter:1.0",
            model_id="some-model",
            prompt_catalog_id="missing",
        )
    with pytest.raises(ValidationError):
        ArtifactRecord(
            source=source,
            canonical_attempts=tuple(reversed(attempts(source))),
            scan_diagnostics=(),
            decode_diagnostics=(),
            metadata=metadata(),
        )
