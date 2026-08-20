"""Regression tests for immutable non-semantic source observations."""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from nlir.artifacts.models import (
    Annotation,
    AnnotationKind,
    ArtifactKind,
    DecodeCodec,
    DecodeProvenance,
    ScanOccurrence,
    ScanReport,
    SourceArtifact,
)
from nlir.contracts.common import SourceSpan
from nlir.contracts.diagnostics import Diagnostic, DiagnosticSeverity


def test_source_artifact_preserves_exact_text_and_content_identity() -> None:
    text = "first\\r\\nnaïve"

    artifact = SourceArtifact.from_text(text, source_name="fixture.md")

    assert artifact.text == text
    assert artifact.artifact_id == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert artifact.kind is ArtifactKind.PHYSICAL
    assert SourceArtifact.content_id(text) == artifact.artifact_id


def test_source_span_uses_end_exclusive_code_point_offsets() -> None:
    artifact = SourceArtifact.from_text("aéz", source_name="fixture.txt")

    span = SourceSpan(artifact_id=artifact.artifact_id, start=1, end=2)

    assert span.extract(artifact.text) == "é"


@pytest.mark.parametrize(
    "payload",
    [
        {"artifact_id": "not-a-hash", "start": 0, "end": 1},
        {"artifact_id": "0" * 64, "start": -1, "end": 1},
        {"artifact_id": "0" * 64, "start": 1, "end": 1},
        {"artifact_id": "0" * 64, "start": "0", "end": 1},
        {"artifact_id": "0" * 64, "start": 0, "end": 1, "extra": True},
    ],
)
def test_source_span_rejects_invalid_or_coerced_shape(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SourceSpan.model_validate(payload)


def test_virtual_artifact_requires_complete_decode_provenance() -> None:
    parent = SourceArtifact.from_text("SGVsbG8=", source_name="parent.txt")
    parent_span = SourceSpan(artifact_id=parent.artifact_id, start=0, end=8)
    provenance = DecodeProvenance(
        parent_artifact_id=parent.artifact_id,
        parent_span=parent_span,
        codec=DecodeCodec.BASE64,
        depth=1,
        chain=(),
    )

    child = SourceArtifact.from_text(
        "Hello",
        source_name=f"virtual://{parent.artifact_id}/base64/0-8",
        kind=ArtifactKind.VIRTUAL,
        decode_provenance=provenance,
    )

    assert child.decode_provenance == provenance
    with pytest.raises(ValidationError):
        SourceArtifact.from_text("Hello", source_name="virtual://bad", kind=ArtifactKind.VIRTUAL)


def test_model_inferred_provenance_requires_model_details() -> None:
    parent = SourceArtifact.from_text("encoded", source_name="parent.txt")
    span = SourceSpan(artifact_id=parent.artifact_id, start=0, end=len(parent.text))

    with pytest.raises(ValidationError):
        DecodeProvenance(
            parent_artifact_id=parent.artifact_id,
            parent_span=span,
            codec=DecodeCodec.MODEL_INFERRED,
            depth=1,
            chain=(),
        )


def test_annotations_and_diagnostics_are_immutable_observations() -> None:
    artifact = SourceArtifact.from_text("https://example.invalid/path", source_name="fixture.md")
    span = SourceSpan(artifact_id=artifact.artifact_id, start=0, end=len(artifact.text))
    annotation = Annotation(
        kind=AnnotationKind.URL,
        span=span,
        normalized_values=("https", "example.invalid"),
    )
    diagnostic = Diagnostic(
        code="decode_invalid",
        severity=DiagnosticSeverity.WARNING,
        message="candidate was not valid base64",
        span=span,
    )

    assert annotation.normalized_values == ("https", "example.invalid")
    assert diagnostic.span == span
    with pytest.raises(ValidationError):
        Annotation.model_validate({**annotation.model_dump(), "raw_text": artifact.text})
    with pytest.raises(ValidationError):
        annotation.kind = AnnotationKind.DOMAIN  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Diagnostic.model_validate({**diagnostic.model_dump(), "extra": "no"})


def test_annotation_normalized_values_are_bounded_and_not_a_metadata_blob() -> None:
    artifact = SourceArtifact.from_text("candidate", source_name="fixture.txt")
    span = SourceSpan(artifact_id=artifact.artifact_id, start=0, end=9)

    with pytest.raises(ValidationError):
        Annotation(kind=AnnotationKind.URL, span=span, normalized_values=("x" * 257,))
    with pytest.raises(ValidationError):
        Annotation.model_validate(
            {"kind": "url", "span": span, "normalized_values": ["https"], "metadata": {}}
        )


def test_scan_report_identity_is_deterministic_and_source_linked() -> None:
    artifact = SourceArtifact.from_text("DEMO_TOKEN", source_name="fixture.txt")
    span = SourceSpan(artifact_id=artifact.artifact_id, start=0, end=10)
    occurrence = ScanOccurrence(
        annotation=Annotation(
            kind=AnnotationKind.ENVIRONMENT_VARIABLE,
            span=span,
            normalized_values=("DEMO_TOKEN",),
        )
    )
    report = ScanReport(
        artifact_id=artifact.artifact_id,
        occurrences=(occurrence,),
        diagnostics=(),
        extractor_version="1.0",
    )

    assert report.stable_id == ScanReport.model_validate(report.model_dump()).stable_id
    assert report.occurrences[0].annotation.span.artifact_id == report.artifact_id
    with pytest.raises(ValidationError):
        ScanReport.model_validate({**report.model_dump(), "opaque_text": "candidate"})
