"""Regression tests for deterministic, observation-only text annotations."""

from __future__ import annotations

from nlir.artifacts.extract import extract_annotations
from nlir.artifacts.models import AnnotationKind, SourceArtifact


def source_artifact(text: str, *, source_name: str = "fixture.txt") -> SourceArtifact:
    return SourceArtifact.from_text(text, source_name=source_name)


def test_extractors_preserve_exact_spans_and_retain_overlaps() -> None:
    text = (
        "See https://api.example.invalid/v1 from $DEMO_TOKEN.\r\n"
        "```sh\r\n"
        "curl https://api.example.invalid/v1\r\n"
        "```\r\n"
        "Read /etc/nlir/config.json and include: fixtures/next.md\r\n"
        "SGVsbG8sIHNhZmUgdGV4dCE= deadbeefcafebabe %48%65%6c%6c%6f\r\n"
        "0123456789abcdefFEDCBA9876543210\r\n"
    )
    annotations = extract_annotations(source_artifact(text, source_name="case.md"))

    observed = [(item.kind, text[item.span.start : item.span.end]) for item in annotations]

    assert (AnnotationKind.URL, "https://api.example.invalid/v1") in observed
    assert (AnnotationKind.DOMAIN, "api.example.invalid") in observed
    assert (AnnotationKind.ENVIRONMENT_VARIABLE, "$DEMO_TOKEN") in observed
    assert (
        AnnotationKind.CODE_FENCE,
        "```sh\r\ncurl https://api.example.invalid/v1\r\n```",
    ) in observed
    assert (AnnotationKind.SHELL_COMMAND, "curl https://api.example.invalid/v1") in observed
    assert (AnnotationKind.FILESYSTEM_PATH, "/etc/nlir/config.json") in observed
    assert (AnnotationKind.EXPLICIT_FILE_REFERENCE, "fixtures/next.md") in observed
    assert (AnnotationKind.BASE64_CANDIDATE, "SGVsbG8sIHNhZmUgdGV4dCE=") in observed
    assert (AnnotationKind.HEX_CANDIDATE, "deadbeefcafebabe") in observed
    assert (AnnotationKind.URL_ENCODED_CANDIDATE, "%48%65%6c%6c%6f") in observed
    assert (AnnotationKind.HIGH_ENTROPY_CANDIDATE, "0123456789abcdefFEDCBA9876543210") in observed

    positions = [
        (item.span.start, item.span.end, item.kind.value, item.normalized_values[:1])
        for item in annotations
    ]
    assert positions == sorted(positions)
    assert annotations == extract_annotations(source_artifact(text, source_name="case.md"))


def test_annotation_metadata_is_bounded_and_non_semantic() -> None:
    text = "https://api.example.invalid/v1 $DEMO_TOKEN"
    annotations = extract_annotations(source_artifact(text))

    url = next(item for item in annotations if item.kind is AnnotationKind.URL)
    environment = next(
        item for item in annotations if item.kind is AnnotationKind.ENVIRONMENT_VARIABLE
    )

    assert url.normalized_values == ("https", "api.example.invalid")
    assert environment.normalized_values == ("DEMO_TOKEN",)
    assert all(
        not hasattr(item, "entity") and not hasattr(item, "operation") for item in annotations
    )


def test_extractors_find_named_files_without_a_path_prefix() -> None:
    text = "Read package.json, then update MEMORY.md and config/live.toml."

    observed = {
        text[item.span.start : item.span.end]
        for item in extract_annotations(source_artifact(text, source_name="SKILLS.md"))
        if item.kind is AnnotationKind.EXPLICIT_FILE_REFERENCE
    }

    assert {"package.json", "MEMORY.md"} <= observed


def test_file_reference_labels_do_not_cross_a_line_into_a_code_fence() -> None:
    text = "Common commands include:\n\n```bash\nnpm run test\n```\n"

    file_references = {
        text[item.span.start : item.span.end]
        for item in extract_annotations(source_artifact(text))
        if item.kind is AnnotationKind.EXPLICIT_FILE_REFERENCE
    }

    assert file_references == set()


def test_extractors_do_not_turn_short_or_low_entropy_tokens_into_entropy_candidates() -> None:
    text = "short-token 00000000000000000000000000000000"

    kinds = [item.kind for item in extract_annotations(source_artifact(text))]

    assert AnnotationKind.HIGH_ENTROPY_CANDIDATE not in kinds
