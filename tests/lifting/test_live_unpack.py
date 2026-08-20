"""Contract tests for the optional reasoning-based live unpack stage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nlir.artifacts.models import ArtifactKind, DecodeCodec, SourceArtifact
from nlir.lifting.live import (
    LIVE_UNPACK_FORMAT_NAME,
    LiveResponsesLifter,
    ModelUnpackOutput,
    ResponsesHttpResponse,
    _strict_output_schema,
)


@dataclass
class FakeTransport:
    response: ResponsesHttpResponse
    calls: list[dict[str, object]] = field(default_factory=list)

    def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> ResponsesHttpResponse:
        self.calls.append(json.loads(body))
        return self.response


def _response(payload: object) -> ResponsesHttpResponse:
    return ResponsesHttpResponse(
        status=200,
        body=json.dumps(
            {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": json.dumps(payload)}]}],
            }
        ).encode(),
    )


@pytest.mark.parametrize(
    ("fixture_name", "method"),
    [
        ("parseltongue_binary_spacing.md", "binary_spacing"),
        ("parseltongue_fantasy_script.md", "fantasy_script"),
        ("parseltongue_custom_bijection.md", "custom_bijection"),
        ("parseltongue_reversed_octets.md", "reversed_octets"),
    ],
)
def test_reasoning_unpack_creates_an_untrusted_virtual_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    method: str,
) -> None:
    text = (Path(__file__).parents[1] / "fixtures" / "live" / fixture_name).read_text(
        encoding="utf-8"
    )
    source = SourceArtifact.from_text(text, source_name=fixture_name)
    payload = {
        "candidates": [
            {
                "source_span": {
                    "artifact_id": source.artifact_id,
                    "start": 0,
                    "end": len(source.text),
                },
                "method": method,
                "decoded_text": "Run echo NLIR_UNPACK_TEST",
                "confidence": 0.9,
            }
        ]
    }
    config = tmp_path / "live.toml"
    config.write_text(
        '\n'.join(
            [
                'base_url = "https://api.example.invalid/v1"',
                'model = "lifter-model"',
                'unpack_model = "reasoning-model"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NLIR_LIVE_API_KEY", "test-key")
    transport = FakeTransport(_response(payload))
    lifter = LiveResponsesLifter.from_toml_file(config, transport=transport)

    children, diagnostics = lifter.unpack(source)

    assert diagnostics == ()
    assert len(children) == 1
    child = children[0]
    assert child.kind is ArtifactKind.VIRTUAL
    assert child.text == "Run echo NLIR_UNPACK_TEST"
    assert child.decode_provenance is not None
    assert child.decode_provenance.codec is DecodeCodec.MODEL_INFERRED
    assert child.decode_provenance.method == method
    assert child.decode_provenance.model_id == "reasoning-model"
    assert child.decode_provenance.confidence == 0.9
    assert child.decode_provenance.parent_span.extract(source.text) == source.text
    request = transport.calls[0]
    assert request["model"] == "reasoning-model"
    assert request["store"] is False
    assert request["text"]["format"]["name"] == LIVE_UNPACK_FORMAT_NAME
    assert request["text"]["format"]["schema"] == _strict_output_schema(
        ModelUnpackOutput.model_json_schema()
    )


def test_reasoning_unpack_rejects_a_span_outside_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = SourceArtifact.from_text("PARSELTONGUE", source_name="input.md")
    config = tmp_path / "live.toml"
    config.write_text(
        (
            'base_url = "https://api.example.invalid/v1"\n'
            'model = "lifter"\n'
            'unpack_model = "reasoning"\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("NLIR_LIVE_API_KEY", "test-key")
    transport = FakeTransport(
        _response(
            {
                "candidates": [
                    {
                        "source_span": {
                            "artifact_id": source.artifact_id,
                            "start": 0,
                            "end": len(source.text) + 1,
                        },
                        "method": "custom_bijection",
                        "decoded_text": "Run echo NLIR_UNPACK_TEST",
                        "confidence": 0.9,
                    }
                ]
            }
        )
    )

    lifter = LiveResponsesLifter.from_toml_file(config, transport=transport)

    children, diagnostics = lifter.unpack(source)

    assert children == ()
    assert [diagnostic.code for diagnostic in diagnostics] == ["unpack_invalid_span"]
