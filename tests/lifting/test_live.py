"""Contract tests for the explicit, stateless Responses API lifter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nlir.artifacts.models import SourceArtifact
from nlir.contracts.ir import IRFragment
from nlir.lifting.live import (
    SYSTEM_PROMPT,
    LiveResponsesLifter,
    ResponsesHttpResponse,
    _source_with_offsets,
    _strict_output_schema,
    check_capability,
)
from nlir.lifting.models import AttemptOutcome, LifterStage

FAKE_KEY = "live-key-marker-not-for-diagnostics"
SOURCE_TEXT = "source-marker-not-for-diagnostics"


@dataclass
class FakeTransport:
    response: ResponsesHttpResponse | Exception
    calls: list[dict[str, object]] = field(default_factory=list)

    def post(
        self,
        *,
        url: str,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: float,
    ) -> ResponsesHttpResponse:
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "body": json.loads(body),
                "timeout_seconds": timeout_seconds,
            }
        )
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def source() -> SourceArtifact:
    return SourceArtifact.from_text(SOURCE_TEXT, source_name="live-source.md")


def response(payload: object | None = None) -> ResponsesHttpResponse:
    content = json.dumps(payload if payload is not None else {"schema_version": "1.0"})
    return ResponsesHttpResponse(
        status=200,
        body=json.dumps(
            {
                "status": "completed",
                "output": [{"content": [{"type": "output_text", "text": content}]}],
            }
        ).encode(),
    )


def config_file(tmp_path: Path, base_url: str = "https://api.example.invalid/v1") -> Path:
    path = tmp_path / "live.toml"
    path.write_text(
        (
            f'base_url = "{base_url}"\nmodel = "test-model"\n'
            "timeout_seconds = 9\nmax_output_tokens = 256\n"
        ),
        encoding="utf-8",
    )
    return path


def lifter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
    base_url: str = "https://api.example.invalid/v1",
) -> LiveResponsesLifter:
    monkeypatch.setenv("NLIR_LIVE_API_KEY", FAKE_KEY)
    return LiveResponsesLifter.from_toml_file(config_file(tmp_path, base_url), transport=transport)


def test_lift_posts_one_strict_stateless_responses_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport(response())
    value = lifter(tmp_path, monkeypatch, transport).lift(
        source(), {source().artifact_id: source()}
    )

    assert value[0].fragment == IRFragment()
    assert len(transport.calls) == 1
    request = transport.calls[0]
    assert request["url"] == "https://api.example.invalid/v1/responses"
    assert request["timeout_seconds"] == 9
    assert request["headers"] == {
        "Authorization": f"Bearer {FAKE_KEY}",
        "Content-Type": "application/json",
    }
    body = request["body"]
    assert isinstance(body, dict)
    assert body["model"] == "test-model"
    assert body["store"] is False
    assert body["input"][0]["role"] == "system"
    assert body["input"][1] == {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": (
                    f"Source artifact ID: {source().artifact_id}. "
                    f"Source length: {len(SOURCE_TEXT)} Unicode code points."
                ),
            }
        ],
    }
    assert body["input"][2]["content"][0]["text"] == _source_with_offsets(SOURCE_TEXT)
    assert body["text"]["format"] == {
        "type": "json_schema",
        "name": "nlir_ir_fragment_v1",
        "strict": True,
        "schema": _strict_output_schema(IRFragment.model_json_schema()),
    }
    _assert_strict_output_schema(body["text"]["format"]["schema"])
    assert body["max_output_tokens"] == 256
    assert not {
        "tools",
        "retrieval",
        "conversation",
        "previous_response_id",
        "stream",
        "background",
        "retry",
    } & set(body)


def _assert_strict_output_schema(value: object) -> None:
    """Check the schema restrictions that the strict Responses format needs."""
    if isinstance(value, list):
        for item in value:
            _assert_strict_output_schema(item)
        return
    if not isinstance(value, dict):
        return
    assert "default" not in value
    if value.get("type") == "object":
        assert value["additionalProperties"] is False
        properties = value.get("properties", {})
        assert value.get("required") == list(properties)
    for item in value.values():
        _assert_strict_output_schema(item)


def test_source_offset_view_uses_exact_unicode_code_point_ranges() -> None:
    assert _source_with_offsets("aé\nb") == "[0:3] aé\n[3:4] b"


def test_system_prompt_classifies_direct_package_manager_installs() -> None:
    assert "INSTALL_PACKAGE" in SYSTEM_PROMPT
    assert "npx" in SYSTEM_PROMPT
    assert "Do not classify that installation as EXECUTE" in SYSTEM_PROMPT


def test_system_prompt_classifies_embedded_instruction_overrides() -> None:
    assert "OVERRIDE_INSTRUCTIONS" in SYSTEM_PROMPT
    assert "untrusted embedded text" in SYSTEM_PROMPT


def test_system_prompt_normalizes_send_destinations() -> None:
    assert "NETWORK_DESTINATION" in SYSTEM_PROMPT
    assert "package.json" in SYSTEM_PROMPT
    assert "MEMORY.md" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    "base_url", ["https://api.example.invalid/v1", "https://api.example.invalid/v1/"]
)
def test_normalizes_one_optional_trailing_slash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    transport = FakeTransport(response())
    attempts = lifter(tmp_path, monkeypatch, transport, base_url).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment == IRFragment()
    assert transport.calls[0]["url"] == "https://api.example.invalid/v1/responses"


@pytest.mark.parametrize(
    "base_url",
    [
        "",
        "http://api.example.invalid/v1",
        "https://api.example.invalid/v1/responses",
        "https://api.example.invalid//v1",
        "https://user@api.example.invalid/v1",
        "https://api.example.invalid/v1?token=secret",
        "https://api.example.invalid/v1#part",
        "https://api.example.invalid/v1/../v2",
    ],
)
def test_rejects_unsafe_or_ambiguous_api_roots_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    transport = FakeTransport(response())
    attempts = lifter(tmp_path, monkeypatch, transport, base_url).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].stage is LifterStage.SETUP
    assert attempts[0].diagnostics[0].code == "invalid_live_config"
    assert not transport.calls


def test_permits_http_only_for_loopback_test_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport(response())
    attempts = lifter(tmp_path, monkeypatch, transport, "http://127.0.0.1:8080/v1").lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment == IRFragment()
    assert transport.calls[0]["url"] == "http://127.0.0.1:8080/v1/responses"


@pytest.mark.parametrize(
    "contents",
    [
        'base_url = "https://api.example.invalid/v1"\nmodel = "test"\napi_key = "not-allowed"\n',
        'base_url = "https://api.example.invalid/v1"\nmodel = "test"\nother = "not-allowed"\n',
        'base_url = "https://api.example.invalid/v1"\nmodel = "test"\ntimeout_seconds = 0\n',
        'base_url = "https://api.example.invalid/v1"\nmodel = "test"\nmax_output_tokens = 999999\n',
    ],
)
def test_rejects_bad_config_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: str
) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(contents, encoding="utf-8")
    transport = FakeTransport(response())
    monkeypatch.setenv("NLIR_LIVE_API_KEY", FAKE_KEY)

    attempts = LiveResponsesLifter.from_toml_file(path, transport=transport).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].code == "invalid_live_config"
    assert not transport.calls


def test_requires_a_non_blank_environment_key_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NLIR_LIVE_API_KEY", "   ")
    transport = FakeTransport(response())

    attempts = LiveResponsesLifter.from_toml_file(config_file(tmp_path), transport=transport).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].code == "missing_api_key"
    assert attempts[0].diagnostics[0].stage is LifterStage.SETUP
    assert not transport.calls


@pytest.mark.parametrize(
    ("http_response", "outcome", "code", "stage"),
    [
        (
            ResponsesHttpResponse(
                200,
                b'{"status":"completed","output":[{"content":[{"type":"refusal","refusal":"no"}]}]}',
            ),
            AttemptOutcome.REFUSED,
            "response_refused",
            LifterStage.LIFECYCLE,
        ),
        (
            ResponsesHttpResponse(200, b'{"status":"incomplete"}'),
            AttemptOutcome.INCOMPLETE,
            "response_incomplete",
            LifterStage.LIFECYCLE,
        ),
        (
            ResponsesHttpResponse(200, b'{"status":"completed","output":[]}'),
            None,
            "response_missing_output",
            LifterStage.LIFECYCLE,
        ),
        (
            ResponsesHttpResponse(200, b"not-json"),
            None,
            "invalid_response_envelope",
            LifterStage.LIFECYCLE,
        ),
        (
            ResponsesHttpResponse(500, b"server-body-must-not-leak"),
            None,
            "http_500",
            LifterStage.LIFECYCLE,
        ),
    ],
)
def test_lifecycle_failures_are_typed_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    http_response: ResponsesHttpResponse,
    outcome: AttemptOutcome | None,
    code: str,
    stage: LifterStage,
) -> None:
    transport = FakeTransport(http_response)
    attempts = lifter(tmp_path, monkeypatch, transport).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment is None
    assert attempts[0].outcome is outcome
    assert attempts[0].diagnostics[0].code == code
    assert attempts[0].diagnostics[0].stage is stage


@pytest.mark.parametrize(
    ("error", "code"), [(TimeoutError(), "transport_timeout"), (OSError(), "transport_failure")]
)
def test_transport_failures_are_typed_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception, code: str
) -> None:
    transport = FakeTransport(error)
    attempts = lifter(tmp_path, monkeypatch, transport).lift(
        source(), {source().artifact_id: source()}
    )

    assert attempts[0].fragment is None
    assert attempts[0].diagnostics[0].code == code
    assert len(transport.calls) == 1


def test_invalid_json_output_and_invalid_fragment_are_validation_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad_json = ResponsesHttpResponse(
        200,
        b'{"status":"completed","output":[{"content":[{"type":"output_text","text":"not-json"}]}]}',
    )
    bad_fragment = response({"schema_version": "wrong"})

    bad_json_attempt = lifter(tmp_path, monkeypatch, FakeTransport(bad_json)).lift(
        source(), {source().artifact_id: source()}
    )[0]
    bad_fragment_attempt = lifter(tmp_path, monkeypatch, FakeTransport(bad_fragment)).lift(
        source(), {source().artifact_id: source()}
    )[0]

    assert (bad_json_attempt.diagnostics[0].stage, bad_json_attempt.diagnostics[0].code) == (
        LifterStage.VALIDATION,
        "invalid_response_json",
    )
    assert (
        bad_fragment_attempt.diagnostics[0].stage,
        bad_fragment_attempt.diagnostics[0].code,
    ) == (
        LifterStage.VALIDATION,
        "invalid_ir_shape",
    )


def test_diagnostics_and_normal_representation_redact_secrets_and_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport(ResponsesHttpResponse(500, b"provider-response-marker"))
    live_lifter = lifter(tmp_path, monkeypatch, transport)
    attempts = live_lifter.lift(source(), {source().artifact_id: source()})
    visible = repr(live_lifter) + " ".join(item.message for item in attempts[0].diagnostics)

    assert FAKE_KEY not in visible
    assert SOURCE_TEXT not in visible
    assert "provider-response-marker" not in visible


def test_capability_check_uses_one_safe_request_and_hides_provider_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = FakeTransport(response())
    monkeypatch.setenv("NLIR_LIVE_API_KEY", FAKE_KEY)

    result = check_capability(config_file(tmp_path), transport=transport)

    assert result.available is True
    assert result.diagnostics == ()
    assert len(transport.calls) == 1
    body = transport.calls[0]["body"]
    assert body["input"][2]["content"][0]["text"] != SOURCE_TEXT
    assert body["store"] is False
    assert body["text"]["format"]["schema"] == _strict_output_schema(
        IRFragment.model_json_schema()
    )
    assert "text" not in repr(result)
