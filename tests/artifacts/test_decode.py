"""Behavior tests for bounded, text-only encoded virtual children."""

from __future__ import annotations

import base64
import builtins
import importlib
import os
import socket
import subprocess

import pytest

from nlir.artifacts.decode import DecodeLimits, decode_artifact
from nlir.artifacts.models import ArtifactKind, DecodeStep, SourceArtifact


def _root(text: str) -> SourceArtifact:
    return SourceArtifact.from_text(text, source_name="fixture.txt")


def _limits(**overrides: int) -> DecodeLimits:
    return DecodeLimits(candidate_minimum_chars=4, **overrides)


@pytest.mark.parametrize(
    ("encoded", "expected"),
    [
        (base64.b64encode(b"decoded base64 text").decode(), "decoded base64 text"),
        (b"decoded hex text".hex(), "decoded hex text"),
        ("%64%65%63%6f%64%65%64%20%75%72%6c%20%74%65%78%74", "decoded url text"),
    ],
)
def test_decodes_each_supported_codec_to_an_inert_virtual_child(
    encoded: str, expected: str
) -> None:
    root = _root(encoded)

    result = decode_artifact(root, limits=_limits())

    assert len(result.children) == 1
    child = result.children[0]
    assert child.artifact.text == expected
    assert child.artifact.kind is ArtifactKind.VIRTUAL
    assert child.artifact.source_name.startswith("virtual://")
    assert child.artifact.decode_provenance is not None
    assert child.artifact.decode_provenance.parent_artifact_id == root.artifact_id
    assert child.artifact.decode_provenance.parent_span.extract(root.text) == encoded
    assert child.annotations == ()
    assert result.diagnostics == ()


def test_nested_children_keep_immediate_parent_and_preceding_chain() -> None:
    inner = base64.b64encode(b"nested text evidence").decode()
    outer = base64.b64encode(inner.encode()).decode()

    result = decode_artifact(_root(outer), limits=_limits(max_depth=2))

    assert [child.artifact.text for child in result.children] == [inner, "nested text evidence"]
    first, second = result.children
    assert second.artifact.decode_provenance is not None
    assert second.artifact.decode_provenance.depth == 2
    assert second.artifact.decode_provenance.parent_artifact_id == first.artifact.artifact_id
    assert second.artifact.decode_provenance.chain == (
        DecodeStep(
            parent_artifact_id=first.artifact.decode_provenance.parent_artifact_id,
            parent_span=first.artifact.decode_provenance.parent_span,
            codec=first.artifact.decode_provenance.codec,
        ),
    )


def test_identical_text_from_different_parent_spans_has_distinct_virtual_identity() -> None:
    encoded = base64.b64encode(b"same decoded text").decode()
    result = decode_artifact(_root(f"{encoded} {encoded}"), limits=_limits())

    assert len(result.children) == 2
    assert result.children[0].artifact.text == result.children[1].artifact.text
    assert result.children[0].artifact.artifact_id != result.children[1].artifact.artifact_id


def test_candidate_order_and_child_annotations_are_deterministic() -> None:
    first = base64.b64encode(b"See https://first.example.invalid").decode()
    second = base64.b64encode(b"Read $DEMO_TOKEN").decode()
    root = _root(f"{second} {first}")

    first_run = decode_artifact(root, limits=_limits())
    second_run = decode_artifact(root, limits=_limits())

    assert first_run == second_run
    assert [child.artifact.text for child in first_run.children] == [
        "Read $DEMO_TOKEN",
        "See https://first.example.invalid",
    ]
    assert [annotation.kind.value for annotation in first_run.children[0].annotations] == [
        "environment_variable"
    ]
    assert [annotation.kind.value for annotation in first_run.children[1].annotations] == [
        "url",
        "domain",
    ]


@pytest.mark.parametrize(
    ("text", "limits", "expected_code"),
    [
        ("/" * 32, _limits(), "decode_non_text"),
        ("0" * 32, _limits(max_child_bytes=8), "decode_limit"),
        (base64.b64encode(b"x" * 12).decode(), _limits(max_candidate_bytes=8), "decode_limit"),
        ("%ff%fe%fd%fc", _limits(), "decode_non_text"),
    ],
)
def test_rejected_candidates_produce_only_stable_diagnostics(
    text: str, limits: DecodeLimits, expected_code: str
) -> None:
    result = decode_artifact(_root(text), limits=limits)

    assert result.children == ()
    assert [diagnostic.code for diagnostic in result.diagnostics] == [expected_code]


def test_aggregate_child_and_depth_budgets_stop_expansion() -> None:
    one = base64.b64encode(b"first child!").decode()
    two = base64.b64encode(b"second child").decode()
    aggregate = decode_artifact(
        _root(f"{one} {two}"), limits=_limits(max_aggregate_bytes=12, max_child_bytes=12)
    )
    children = decode_artifact(_root(f"{one} {two}"), limits=_limits(max_children=1))
    nested = base64.b64encode(one.encode()).decode()
    depth = decode_artifact(_root(nested), limits=_limits(max_depth=1))

    assert len(aggregate.children) == 1
    assert [item.code for item in aggregate.diagnostics] == ["decode_limit"]
    assert len(children.children) == 1
    assert [item.code for item in children.diagnostics] == ["decode_limit"]
    assert len(depth.children) == 1
    assert [item.code for item in depth.diagnostics] == ["decode_limit"]


def test_decoding_does_not_invoke_operational_side_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    encoded = base64.b64encode(b"safe decoded text").decode()
    invoked: list[str] = []

    def forbidden(name: str):
        def fail(*_args: object, **_kwargs: object) -> None:
            invoked.append(name)
            raise AssertionError(f"unexpected side effect: {name}")

        return fail

    monkeypatch.setattr(subprocess, "run", forbidden("subprocess"))
    monkeypatch.setattr(os, "system", forbidden("shell"))
    monkeypatch.setattr(socket, "create_connection", forbidden("network"))
    monkeypatch.setattr(importlib, "import_module", forbidden("dynamic_import"))
    monkeypatch.setattr(builtins, "open", forbidden("file_write"))

    result = decode_artifact(_root(encoded), limits=_limits())

    assert len(result.children) == 1
    assert invoked == []
