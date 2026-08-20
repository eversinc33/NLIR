"""Shared test builders for in-memory canonical source artifacts."""

from __future__ import annotations


def source_artifact(text: str, *, source_name: str = "fixture.txt"):
    """Build a physical source artifact without touching the filesystem."""
    from nlir.artifacts.models import SourceArtifact

    return SourceArtifact.from_text(text, source_name=source_name)
