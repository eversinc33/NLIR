"""Regression tests for deterministic, fault-tolerant directory scans."""

from __future__ import annotations

from nlir.artifacts.loader import scan_directory


def test_directory_scan_is_path_sorted_keeps_duplicate_content_and_continues(tmp_path) -> None:
    (tmp_path / "z.txt").write_text("same", encoding="utf-8")
    (tmp_path / "a.md").write_text("same", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.YML").write_text("value: naïve\r\n", encoding="utf-8", newline="")
    (tmp_path / "ignored.bin").write_bytes(b"not scanned")
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    (tmp_path / "link.txt").symlink_to(tmp_path / "a.md")

    result = scan_directory(tmp_path)

    assert [item.relative_path for item in result.artifacts] == ["a.md", "nested/b.YML", "z.txt"]
    assert result.artifacts[0].artifact.artifact_id == result.artifacts[2].artifact.artifact_id
    assert [item.path for item in result.diagnostics] == ["bad.txt", "ignored.bin", "link.txt"]
    assert [item.code for item in result.diagnostics] == [
        "invalid_utf8",
        "unsupported_suffix",
        "skipped_symlink",
    ]


def test_directory_scan_uses_posix_lexical_order_regardless_of_creation_order(tmp_path) -> None:
    for name in ("B.txt", "a.txt", "A.txt"):
        (tmp_path / name).write_text(name, encoding="utf-8")

    assert [item.relative_path for item in scan_directory(tmp_path).artifacts] == [
        "A.txt",
        "B.txt",
        "a.txt",
    ]
