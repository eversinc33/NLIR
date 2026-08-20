"""Regression tests for strict, canonical physical artifact loading."""

from __future__ import annotations

import hashlib

import pytest

from nlir.artifacts.loader import LoadFailure, load_file


def test_load_file_preserves_crlf_bom_and_unicode_without_normalizing(tmp_path) -> None:
    path = tmp_path / "instruction.MD"
    text = "\ufefffirst\r\nnaïve"
    path.write_bytes(text.encode("utf-8"))

    loaded = load_file(path)

    assert loaded.artifact.text == text
    assert loaded.artifact.artifact_id == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert loaded.relative_path == "instruction.MD"
    assert loaded.artifact.text[10:11] == "ï"


@pytest.mark.parametrize(
    ("name", "payload", "code"),
    [
        ("unsupported.pdf", b"plain text", "unsupported_suffix"),
        ("bad.txt", b"\xff", "invalid_utf8"),
        ("large.txt", b"x" * (1024 * 1024 + 1), "oversized_file"),
    ],
)
def test_load_file_rejects_unsupported_oversized_and_invalid_utf8(
    tmp_path, name: str, payload: bytes, code: str
) -> None:
    path = tmp_path / name
    path.write_bytes(payload)

    with pytest.raises(LoadFailure) as raised:
        load_file(path)

    assert raised.value.diagnostic.code == code
    assert raised.value.diagnostic.path == name


def test_load_file_never_ingests_symlink(tmp_path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe text", encoding="utf-8")
    link = tmp_path / "linked.txt"
    link.symlink_to(target)

    with pytest.raises(LoadFailure) as raised:
        load_file(link)

    assert raised.value.diagnostic.code == "skipped_symlink"
