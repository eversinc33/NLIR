"""Small, deterministic text observations with exact source spans.

These extractors intentionally describe syntax only.  They never resolve a
reference, decode a candidate, or create an IR entity, operation, or relation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable

from nlir.artifacts.models import Annotation, AnnotationKind, SourceArtifact
from nlir.contracts.common import SourceSpan

_URL_RE = re.compile(r"https?://[^\s<>\"'`()\[\]{}]+", re.IGNORECASE)
_DOMAIN_RE = re.compile(
    r"(?<![A-Za-z0-9_-])(?:[A-Za-z0-9-]+\.)+[A-Za-z][A-Za-z0-9-]*(?![A-Za-z0-9_-])"
)
_IP_RE = re.compile(r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])")
_PATH_RE = re.compile(r"(?<![A-Za-z0-9_:/])(?:~[\\/]|[A-Za-z]:\\|\.{1,2}/|/)[^\s<>\"'`()\[\]{}]+")
_ENV_RE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
_CODE_FENCE_RE = re.compile(r"```[^\r\n]*\r?\n.*?```", re.DOTALL)
_SHELL_LINE_RE = re.compile(
    r"(?m)^\s*(?:[$#]\s*)?(?:curl|wget|python(?:3)?|node|bash|sh|zsh|pwsh|powershell|"
    r"rm|cp|mv|cat|grep|sed|awk|chmod|chown|git)\b[^\r\n]*"
)
_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/]{8,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_RE = re.compile(r"(?<![A-Fa-f0-9])(?:[A-Fa-f0-9]{2}){4,}(?![A-Fa-f0-9])")
_PERCENT_RE = re.compile(r"(?<!%)(?:%[0-9A-Fa-f]{2}){4,}")
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9+/=_-]{32,}")
_FILE_REFERENCE_RE = re.compile(
    r"(?i)\b(?:include|file|path|reference)[ \t]*:[ \t]*(?P<reference>[^\s#]+)"
)
_FILE_NAME_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9][A-Za-z0-9_.-]{0,127})\."
    r"(?:md|txt|json|yaml|yml|toml|ini|cfg|conf|lock|lockb|py|js|ts|tsx|jsx|sh|ps1)"
    r"(?![A-Za-z0-9_-]|\.[A-Za-z0-9])"
)


def extract_annotations(artifact: SourceArtifact) -> tuple[Annotation, ...]:
    """Return all source-linked syntactic observations in stable span order."""
    annotations: list[Annotation] = []
    annotations.extend(_matches(artifact, _URL_RE, AnnotationKind.URL, _url_values))
    annotations.extend(_matches(artifact, _DOMAIN_RE, AnnotationKind.DOMAIN, _domain_values))
    annotations.extend(_valid_ip_matches(artifact))
    annotations.extend(_matches(artifact, _PATH_RE, AnnotationKind.FILESYSTEM_PATH))
    annotations.extend(_environment_matches(artifact))
    annotations.extend(_matches(artifact, _CODE_FENCE_RE, AnnotationKind.CODE_FENCE))
    annotations.extend(_matches(artifact, _SHELL_LINE_RE, AnnotationKind.SHELL_COMMAND))
    annotations.extend(_matches(artifact, _BASE64_RE, AnnotationKind.BASE64_CANDIDATE))
    annotations.extend(_matches(artifact, _HEX_RE, AnnotationKind.HEX_CANDIDATE))
    annotations.extend(_matches(artifact, _PERCENT_RE, AnnotationKind.URL_ENCODED_CANDIDATE))
    annotations.extend(_high_entropy_matches(artifact))
    annotations.extend(_file_reference_matches(artifact))
    annotations.extend(_matches(artifact, _FILE_NAME_RE, AnnotationKind.EXPLICIT_FILE_REFERENCE))
    return tuple(
        sorted(
            annotations,
            key=lambda item: (
                item.span.start,
                item.span.end,
                item.kind.value,
                item.normalized_values[0] if item.normalized_values else "",
            ),
        )
    )


def _matches(
    artifact: SourceArtifact,
    pattern: re.Pattern[str],
    kind: AnnotationKind,
    values: Callable[[re.Match[str]], tuple[str, ...]] | None = None,
) -> Iterable[Annotation]:
    for match in pattern.finditer(artifact.text):
        yield _annotation(
            artifact, kind, match.start(), match.end(), values(match) if values else ()
        )


def _url_values(match: re.Match[str]) -> tuple[str, ...]:
    value = match.group(0)
    scheme, remaining = value.split("://", 1)
    return (scheme.casefold(), remaining.split("/", 1)[0].casefold())


def _domain_values(match: re.Match[str]) -> tuple[str, ...]:
    return (match.group(0).casefold(),)


def _valid_ip_matches(artifact: SourceArtifact) -> Iterable[Annotation]:
    for match in _IP_RE.finditer(artifact.text):
        if all(int(part) <= 255 for part in match.group(0).split(".")):
            yield _annotation(
                artifact, AnnotationKind.IP_ADDRESS, match.start(), match.end(), (match.group(0),)
            )


def _environment_matches(artifact: SourceArtifact) -> Iterable[Annotation]:
    for match in _ENV_RE.finditer(artifact.text):
        name = match.group("braced") or match.group("plain")
        yield _annotation(
            artifact, AnnotationKind.ENVIRONMENT_VARIABLE, match.start(), match.end(), (name,)
        )


def _high_entropy_matches(artifact: SourceArtifact) -> Iterable[Annotation]:
    for match in _ASCII_TOKEN_RE.finditer(artifact.text):
        token = match.group(0)
        if _shannon_entropy(token) >= 3.5:
            yield _annotation(
                artifact, AnnotationKind.HIGH_ENTROPY_CANDIDATE, match.start(), match.end()
            )


def _file_reference_matches(artifact: SourceArtifact) -> Iterable[Annotation]:
    for match in _FILE_REFERENCE_RE.finditer(artifact.text):
        start, end = match.span("reference")
        yield _annotation(
            artifact,
            AnnotationKind.EXPLICIT_FILE_REFERENCE,
            start,
            end,
            (match.group("reference"),),
        )


def _annotation(
    artifact: SourceArtifact,
    kind: AnnotationKind,
    start: int,
    end: int,
    normalized_values: tuple[str, ...] = (),
) -> Annotation:
    return Annotation(
        kind=kind,
        span=SourceSpan(artifact_id=artifact.artifact_id, start=start, end=end),
        normalized_values=normalized_values,
    )


def _shannon_entropy(token: str) -> float:
    """Calculate character entropy for one already-bounded ASCII token."""
    length = len(token)
    return -sum((count / length) * math.log2(count / length) for count in Counter(token).values())
