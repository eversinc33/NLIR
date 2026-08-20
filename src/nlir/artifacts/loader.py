"""Strict, bounded, deterministic loading of physical text artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nlir.artifacts.decode import DecodeResult, decode_artifact
from nlir.artifacts.models import DecodeLimits, SourceArtifact

SUPPORTED_SUFFIXES = frozenset({".md", ".txt", ".yaml", ".yml", ".json"})
MAX_SOURCE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class PathDiagnostic:
    """A stable, path-linked scanner diagnostic for an unscannable input."""

    path: str
    code: str
    message: str


class LoadFailure(Exception):
    """Raised for an individual input that cannot be admitted as source text."""

    def __init__(self, diagnostic: PathDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """One physical source plus its path occurrence within the requested scan."""

    artifact: SourceArtifact
    relative_path: str


@dataclass(frozen=True, slots=True)
class DirectoryScan:
    """Sorted physical artifacts and recoverable path diagnostics."""

    artifacts: tuple[LoadedArtifact, ...]
    diagnostics: tuple[PathDiagnostic, ...]


@dataclass(frozen=True, slots=True)
class ScannedArtifact:
    """One loaded root plus its bounded, inert virtual-child scan result."""

    loaded: LoadedArtifact
    decoded: DecodeResult


def scan_loaded_artifact(
    loaded: LoadedArtifact, *, decode_limits: DecodeLimits | None = None
) -> ScannedArtifact:
    """Apply the pure extractor/decoder path to one already-loaded root artifact."""
    return ScannedArtifact(
        loaded=loaded,
        decoded=decode_artifact(loaded.artifact, limits=decode_limits),
    )


def load_file(path: Path, *, relative_path: str | None = None) -> LoadedArtifact:
    """Load one supported non-symlink file without changing its decoded text."""
    display_path = relative_path or path.name
    if path.is_symlink():
        raise LoadFailure(
            _diagnostic(display_path, "skipped_symlink", "symlink inputs are not scanned")
        )
    if not path.is_file():
        raise LoadFailure(
            _diagnostic(display_path, "invalid_target", "target is not a regular file")
        )
    if path.suffix.casefold() not in SUPPORTED_SUFFIXES:
        raise LoadFailure(
            _diagnostic(
                display_path, "unsupported_suffix", "file suffix is not supported for text scans"
            )
        )

    try:
        with path.open("rb") as source_file:
            payload = source_file.read(MAX_SOURCE_BYTES + 1)
    except OSError as error:
        raise LoadFailure(_diagnostic(display_path, "unreadable_file", str(error))) from error

    if len(payload) > MAX_SOURCE_BYTES:
        raise LoadFailure(
            _diagnostic(display_path, "oversized_file", "file exceeds the 1 MiB source limit")
        )
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise LoadFailure(_diagnostic(display_path, "invalid_utf8", str(error))) from error

    return LoadedArtifact(
        artifact=SourceArtifact.from_text(text, source_name=display_path),
        relative_path=display_path,
    )


def scan_directory(path: Path) -> DirectoryScan:
    """Walk a directory by POSIX lexical path, retaining recoverable failures."""
    if path.is_symlink() or not path.is_dir():
        raise LoadFailure(_diagnostic(path.name, "invalid_target", "target is not a directory"))

    loaded: list[LoadedArtifact] = []
    diagnostics: list[PathDiagnostic] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        relative_path = candidate.relative_to(path).as_posix()
        if candidate.is_dir() and not candidate.is_symlink():
            continue
        try:
            loaded.append(load_file(candidate, relative_path=relative_path))
        except LoadFailure as error:
            diagnostics.append(error.diagnostic)

    return DirectoryScan(artifacts=tuple(loaded), diagnostics=tuple(diagnostics))


def _diagnostic(path: str, code: str, message: str) -> PathDiagnostic:
    return PathDiagnostic(path=path, code=code, message=message)
