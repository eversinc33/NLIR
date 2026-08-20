"""Deterministic promotion of exact source indicators into IR entities."""

from __future__ import annotations

import hashlib
import ipaddress
import re
from urllib.parse import urlsplit

from nlir.artifacts.extract import extract_annotations
from nlir.artifacts.models import Annotation, AnnotationKind, SourceArtifact
from nlir.contracts.ir import Entity, EntityType, Sensitivity, TrustLevel

_ENTITY_TYPES = {
    AnnotationKind.URL: EntityType.NETWORK_DESTINATION,
    AnnotationKind.DOMAIN: EntityType.NETWORK_DESTINATION,
    AnnotationKind.IP_ADDRESS: EntityType.NETWORK_DESTINATION,
    AnnotationKind.FILESYSTEM_PATH: EntityType.FILE,
    AnnotationKind.EXPLICIT_FILE_REFERENCE: EntityType.FILE,
    AnnotationKind.ENVIRONMENT_VARIABLE: EntityType.ENVIRONMENT_VARIABLE,
}
_DOMAIN_SUFFIXES = frozenset(
    {
        "ai", "app", "at", "au", "be", "biz", "br", "ca", "ch", "cloud", "cn", "co",
        "com", "cz", "de", "dev", "dk", "edu", "es", "eu", "fi", "fr", "gov", "in",
        "info", "int", "io", "it", "jp", "kr", "me", "mil", "mx", "net", "nl", "no",
        "nz", "online", "org", "pl", "pro", "ru", "se", "sh", "site", "store", "tech",
        "tv", "ua", "uk", "us", "xyz", "za",
    }
)
_FILE_ACTION_RE = re.compile(
    r"(?i)\b(?:append|create|delete|edit|inspect|load|modify|open|read|replace|save|update|write)"
    r"(?:[ \t]+(?:a|an|the))?[ \t]*$"
)
_FILE_LABEL_RE = re.compile(r"(?i)\b(?:include|file|path|reference)[ \t]*:[ \t]*$")
_LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


def static_entities(artifact: SourceArtifact) -> tuple[Entity, ...]:
    """Return exact, non-semantic IR entities for supported source indicators."""
    entities: list[Entity] = []
    seen: set[tuple[EntityType, str, int, int]] = set()
    annotations = extract_annotations(artifact)
    for annotation in annotations:
        if (
            annotation.kind is AnnotationKind.DOMAIN
            and _contained_by_higher_signal(annotation, annotations)
        ):
            continue
        entity_type = _ENTITY_TYPES.get(annotation.kind)
        if entity_type is None:
            continue
        value = _value(annotation, artifact)
        if annotation.kind is AnnotationKind.DOMAIN and not _is_supported_domain(value):
            continue
        if (
            annotation.kind is AnnotationKind.EXPLICIT_FILE_REFERENCE
            and not _is_clear_file_reference(annotation, artifact)
        ):
            continue
        if not value or len(value) > 256 or any(character.isspace() for character in value):
            continue
        key = (entity_type, value, annotation.span.start, annotation.span.end)
        if key in seen:
            continue
        seen.add(key)
        entities.append(
            Entity(
                id=_entity_id(entity_type, value, annotation.span.start, annotation.span.end),
                type=entity_type,
                value=value,
                sensitivity=Sensitivity.UNKNOWN,
                trust=_trust(annotation, value),
                evidence=(annotation.span,),
                confidence=1.0,
                underspecified=False,
            )
        )
    return tuple(entities)


def _contained_by_higher_signal(
    annotation: Annotation, annotations: tuple[Annotation, ...]
) -> bool:
    """Skip a domain when a URL or named file already contains the same text."""
    return any(
        other.kind in {AnnotationKind.URL, AnnotationKind.EXPLICIT_FILE_REFERENCE}
        and other.span.start <= annotation.span.start
        and other.span.end >= annotation.span.end
        for other in annotations
    )


def _is_supported_domain(value: str) -> bool:
    """Return true only when a bare domain has one reviewed public suffix."""
    return value.rpartition(".")[2].casefold() in _DOMAIN_SUFFIXES


def _is_clear_file_reference(annotation: Annotation, artifact: SourceArtifact) -> bool:
    """Require a file action, explicit label, or inline code for a bare file name."""
    line_start = artifact.text.rfind("\n", 0, annotation.span.start) + 1
    prefix = artifact.text[line_start : annotation.span.start]
    if _FILE_ACTION_RE.search(prefix) or _FILE_LABEL_RE.search(prefix):
        return True
    marker = chr(96)
    before = artifact.text.rfind(marker, line_start, annotation.span.start)
    after = artifact.text.find(marker, annotation.span.end)
    return before >= line_start and after >= annotation.span.end


def _value(annotation: Annotation, artifact: SourceArtifact) -> str:
    """Use the scanner-normalized value when it safely preserves one indicator."""
    if annotation.kind is AnnotationKind.ENVIRONMENT_VARIABLE:
        return annotation.normalized_values[0]
    value = annotation.span.extract(artifact.text)
    if annotation.kind is AnnotationKind.URL:
        return value.rstrip(".,;!?")
    return value


def _trust(annotation: Annotation, value: str) -> TrustLevel:
    """Classify a literal network target as external unless it is local."""
    if annotation.kind not in {
        AnnotationKind.URL,
        AnnotationKind.DOMAIN,
        AnnotationKind.IP_ADDRESS,
    }:
        return TrustLevel.UNKNOWN
    return TrustLevel.UNKNOWN if _is_local_destination(value) else TrustLevel.EXTERNAL


def _is_local_destination(value: str) -> bool:
    """Return true only for a localhost name or a local IP address."""
    host = urlsplit(value).hostname if "://" in value else value
    if host is None:
        return False
    normalized = host.rstrip(".").casefold()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return any(address in network for network in _LOCAL_NETWORKS)


def _entity_id(entity_type: EntityType, value: str, start: int, end: int) -> str:
    """Build one readable stable source-indicator identifier."""
    material = f"{entity_type.value}\0{value}\0{start}\0{end}".encode()
    suffix = hashlib.sha256(material).hexdigest()[:12]
    return f"static.{entity_type.value.lower()}.{suffix}"
