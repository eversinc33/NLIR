"""Compose source scanning, canonical lifting, IR records, and offline hunts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping

from nlir.artifacts.decode import decode_artifact
from nlir.artifacts.loader import LoadedArtifact, scan_loaded_artifact
from nlir.artifacts.models import DecodeLimits, SourceArtifact
from nlir.artifacts.static_ir import static_entities
from nlir.canonical.models import CanonicalEntity, CanonicalFragment, CanonicalRelationship
from nlir.canonical.normalize import normalize_fragment
from nlir.contracts.common import SourceSpan
from nlir.contracts.ir import EntityType, IRFragment, RelationType
from nlir.ir.models import (
    ArtifactRecord,
    HuntReport,
    HuntResult,
    LiftDiagnostic,
    LiftRecordMetadata,
    SourceLocationHint,
)
from nlir.lifting.canonical import canonicalize_attempts
from nlir.lifting.models import (
    CanonicalAttemptResult,
    CanonicalAttemptStage,
    SemanticLifter,
    SemanticUnpacker,
)
from nlir.rules.evaluate import evaluate_rule
from nlir.rules.models import Rule


def lift_loaded_artifact(
    loaded: LoadedArtifact,
    *,
    lifter: SemanticLifter,
    metadata: LiftRecordMetadata,
    decode_limits: DecodeLimits | None = None,
) -> tuple[ArtifactRecord, ...]:
    """Lift one loaded root and every child artifact into IR records."""
    scanned = scan_loaded_artifact(loaded, decode_limits=decode_limits)
    static_sources = (
        scanned.loaded.artifact,
        *(child.artifact for child in scanned.decoded.children),
    )
    unpack_diagnostics: dict[str, tuple] = {}
    model_children: list[SourceArtifact] = []
    if isinstance(lifter, SemanticUnpacker):
        for source in static_sources:
            children, diagnostics = lifter.unpack(source)
            unpack_diagnostics[source.artifact_id] = diagnostics
            model_children.extend(children)
    model_decode_diagnostics: dict[str, tuple] = {}
    model_static_children: list[SourceArtifact] = []
    for source in model_children:
        decoded = decode_artifact(source, limits=decode_limits)
        model_decode_diagnostics[source.artifact_id] = decoded.diagnostics
        model_static_children.extend(child.artifact for child in decoded.children)
    sources = tuple(
        {
            source.artifact_id: source
            for source in (*static_sources, *model_children, *model_static_children)
        }.values()
    )
    registry = {source.artifact_id: source for source in sources}
    records = tuple(
        ArtifactRecord(
            source=source,
            canonical_attempts=_enrich_static_entities(
                canonicalize_attempts(lifter.lift(source, registry), source, registry),
                source,
                registry,
            ),
            scan_diagnostics=(),
            decode_diagnostics=(
                *(scanned.decoded.diagnostics if source == static_sources[0] else ()),
                *unpack_diagnostics.get(source.artifact_id, ()),
                *model_decode_diagnostics.get(source.artifact_id, ()),
            ),
            metadata=metadata,
        )
        for source in sources
    )
    return _resolve_decoded_references(records, registry)


def _enrich_static_entities(
    attempts: tuple[CanonicalAttemptResult, ...],
    source: SourceArtifact,
    registry: Mapping[str, SourceArtifact],
) -> tuple[CanonicalAttemptResult, ...]:
    """Add exact static indicator entities to each accepted semantic fragment."""
    entities = static_entities(source)
    if not entities:
        return attempts
    normalized = normalize_fragment(IRFragment(entities=entities), source.artifact_id, registry)
    if normalized.fragment is None:
        return attempts
    return tuple(
        attempt
        if attempt.canonical_fragment is None
        else attempt.model_copy(
            update={
                "canonical_fragment": _merge_static_entities(
                    attempt.canonical_fragment, normalized.fragment
                )
            }
        )
        for attempt in attempts
    )


def _merge_static_entities(
    semantic: CanonicalFragment, static: CanonicalFragment
) -> CanonicalFragment:
    """Keep semantic properties and add static evidence for matching identifiers."""
    entities: dict[str, CanonicalEntity] = {entity.id: entity for entity in semantic.entities}
    for static_entity in static.entities:
        current = entities.get(static_entity.id)
        if current is None:
            entities[static_entity.id] = static_entity
            continue
        evidence = tuple(
            sorted(
                {*current.evidence, *static_entity.evidence},
                key=lambda span: (span.artifact_id, span.start, span.end),
            )
        )
        update: dict[str, object] = {"evidence": evidence}
        if (
            static_entity.type.value == "NETWORK_DESTINATION"
            and static_entity.trust.value == "EXTERNAL"
        ):
            update["trust"] = static_entity.trust
        entities[static_entity.id] = current.model_copy(update=update)
    merged = tuple(sorted(entities.values(), key=lambda item: item.id))
    return semantic.model_copy(update={"entities": merged})


def _resolve_decoded_references(
    records: tuple[ArtifactRecord, ...],
    registry: Mapping[str, SourceArtifact],
) -> tuple[ArtifactRecord, ...]:
    """Link an entity that names a still-encoded blob to its decoded resolution.

    Every artifact is decoded before any artifact is semantically lifted, so
    a decoded child's relationship to its parent span is a known pipeline
    fact by the time this runs, not something the model has to notice on its
    own. When some entity in a fragment carries, as its value, the exact
    text of a span that a sibling artifact is a decoded child of, a
    DECODES_TO relationship connects that entity to the child's own resolved
    entity, and the resolved entity (with its own evidence, from the child
    artifact) is copied into the fragment.
    """
    resolved_fragment_by_artifact: dict[str, CanonicalFragment] = {}
    for record in records:
        fragment = _first_accepted_fragment(record)
        if fragment is not None:
            resolved_fragment_by_artifact[record.source.artifact_id] = fragment

    children_by_parent: dict[str, list[SourceArtifact]] = defaultdict(list)
    for source in registry.values():
        provenance = source.decode_provenance
        if provenance is not None:
            children_by_parent[provenance.parent_artifact_id].append(source)

    updated: list[ArtifactRecord] = []
    for record in records:
        children = children_by_parent.get(record.source.artifact_id, ())
        if not children:
            updated.append(record)
            continue
        updated.append(
            record.model_copy(
                update={
                    "canonical_attempts": tuple(
                        _resolve_attempt(
                            attempt, record.source, children, resolved_fragment_by_artifact
                        )
                        for attempt in record.canonical_attempts
                    )
                }
            )
        )
    return tuple(updated)


def _first_accepted_fragment(record: ArtifactRecord) -> CanonicalFragment | None:
    for attempt in record.canonical_attempts:
        if attempt.stage is CanonicalAttemptStage.ACCEPTED and attempt.canonical_fragment:
            return attempt.canonical_fragment
    return None


def _resolve_attempt(
    attempt: CanonicalAttemptResult,
    parent: SourceArtifact,
    children: list[SourceArtifact],
    resolved_fragment_by_artifact: dict[str, CanonicalFragment],
) -> CanonicalAttemptResult:
    if attempt.canonical_fragment is None:
        return attempt
    fragment = attempt.canonical_fragment
    entities: dict[str, CanonicalEntity] = {entity.id: entity for entity in fragment.entities}
    relationships: dict[str, CanonicalRelationship] = {
        relationship.id: relationship for relationship in fragment.relationships
    }
    changed = False
    for child in children:
        provenance = child.decode_provenance
        assert provenance is not None
        span = provenance.parent_span
        candidate = parent.text[span.start : span.end]
        child_fragment = resolved_fragment_by_artifact.get(child.artifact_id)
        if child_fragment is None:
            continue
        full_child_span = (SourceSpan(artifact_id=child.artifact_id, start=0, end=len(child.text)),)
        for raw_entity in list(entities.values()):
            if raw_entity.value != candidate:
                continue
            if raw_entity.type is EntityType.ENCODED_DATA:
                # A still-encoded blob carries no semantic type of its own, so match the
                # one child entity that stands for the whole decoded text instead of
                # requiring a type it can never share.
                matches = [
                    resolved
                    for resolved in child_fragment.entities
                    if resolved.evidence == full_child_span
                ]
            else:
                matches = [
                    resolved
                    for resolved in child_fragment.entities
                    if resolved.type == raw_entity.type
                ]
            if len(matches) != 1:
                continue
            resolved = matches[0]
            entities[resolved.id] = resolved
            relationship_id = _decodes_to_id(raw_entity.id, resolved.id)
            relationships[relationship_id] = CanonicalRelationship(
                id=relationship_id,
                source=raw_entity.id,
                relation=RelationType.DECODES_TO,
                target=resolved.id,
                evidence=resolved.evidence,
                confidence=min(raw_entity.confidence, resolved.confidence),
                underspecified=False,
            )
            changed = True
    if not changed:
        return attempt
    updated_fragment = fragment.model_copy(
        update={
            "entities": tuple(sorted(entities.values(), key=lambda item: item.id)),
            "relationships": tuple(sorted(relationships.values(), key=lambda item: item.id)),
        }
    )
    return attempt.model_copy(update={"canonical_fragment": updated_fragment})


def _decodes_to_id(source_id: str, target_id: str) -> str:
    digest = hashlib.sha256(f"{source_id}\x00{target_id}".encode()).hexdigest()[:16]
    return f"relationship.decodes_to.{digest}"


def hunt_records(records: tuple[ArtifactRecord, ...], rule: Rule) -> HuntReport:
    """Evaluate one rule over the accepted attempts in these IR records."""
    source_registry = {record.source.artifact_id: record.source for record in records}
    usable: list[tuple[str, int, CanonicalFragment]] = []
    extra_diagnostics: list[LiftDiagnostic] = []
    for record in records:
        for attempt in record.canonical_attempts:
            if attempt.stage is not CanonicalAttemptStage.ACCEPTED:
                continue
            fragment = attempt.canonical_fragment
            if fragment is None:
                continue
            invalid_span = _invalid_evidence_span(fragment, source_registry)
            if invalid_span is not None:
                extra_diagnostics.append(
                    LiftDiagnostic(
                        code="invalid_canonical_evidence",
                        message="Canonical evidence does not match its source text.",
                        artifact_id=record.source.artifact_id,
                    )
                )
                continue
            usable.append((record.source.artifact_id, attempt.ordinal, fragment))

    results = tuple(
        _hunt_fragment(rule, artifact_id, ordinal, fragment, source_registry)
        for artifact_id, ordinal, fragment in sorted(usable, key=lambda item: item[:2])
    )
    return HuntReport(results=results, diagnostics=tuple(extra_diagnostics))


def _hunt_fragment(
    rule: Rule,
    artifact_id: str,
    ordinal: int,
    fragment: CanonicalFragment,
    source_registry: Mapping[str, SourceArtifact],
) -> HuntResult:
    """Evaluate one fragment and render its exact evidence only after a hit."""
    result = evaluate_rule(rule, fragment, source_registry[artifact_id])
    if result.status == "NO_HIT":
        return HuntResult(artifact_id=artifact_id, attempt_ordinal=ordinal, status="NO_HIT")
    spans = tuple(span for matched in result.evidence for span in matched.spans)
    return HuntResult(
        artifact_id=artifact_id,
        attempt_ordinal=ordinal,
        status="HIT",
        hints=_location_hints(spans, source_registry),
    )


def _invalid_evidence_span(
    fragment: CanonicalFragment, sources: Mapping[str, SourceArtifact]
) -> SourceSpan | None:
    """Return the first evidence span that its source text cannot support."""
    records = (*fragment.entities, *fragment.operations, *fragment.relationships)
    for record in records:
        for span in record.evidence:
            source = sources.get(span.artifact_id)
            if source is None or span.end > len(source.text):
                return span
    return None


def _location_hints(
    spans: tuple[SourceSpan, ...], sources: Mapping[str, SourceArtifact]
) -> tuple[SourceLocationHint, ...]:
    """Build ordered locations and map virtual evidence to its input source."""
    hints: dict[tuple[str, str, int, int], SourceLocationHint] = {}
    for span in spans:
        source_span = resolve_input_source_span(span, sources)
        source = sources[source_span.artifact_id]
        key = (source.source_name, source_span.artifact_id, source_span.start, source_span.end)
        hints[key] = SourceLocationHint(
            artifact_id=source_span.artifact_id,
            source_name=source.source_name,
            start=source_span.start,
            end=source_span.end,
            line=_line_number(source.text, source_span.start),
            column=_column_number(source.text, source_span.start),
        )
    return tuple(hints[key] for key in sorted(hints))


def resolve_input_source_span(
    span: SourceSpan, sources: Mapping[str, SourceArtifact]
) -> SourceSpan:
    """Return the original input span for virtual evidence when its parent is present."""
    current = span
    visited: set[str] = set()
    while current.artifact_id not in visited:
        visited.add(current.artifact_id)
        source = sources[current.artifact_id]
        provenance = source.decode_provenance
        if provenance is None or provenance.parent_artifact_id not in sources:
            return current
        current = provenance.parent_span
    return span


def _line_number(text: str, offset: int) -> int:
    """Return the one-based line for a Python code-point offset."""
    return text.count("\n", 0, offset) + 1


def _column_number(text: str, offset: int) -> int:
    """Return the one-based code-point column without text normalization."""
    return offset - text.rfind("\n", 0, offset)
