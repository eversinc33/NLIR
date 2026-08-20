"""Tests for deterministic source-indicator IR entities."""

from __future__ import annotations

from nlir.artifacts.models import SourceArtifact
from nlir.artifacts.static_ir import static_entities
from nlir.contracts.ir import EntityType, TrustLevel


def test_static_entities_promote_files_network_indicators_and_environment_names() -> None:
    source = SourceArtifact.from_text(
        "Read package.json and $DEMO_TOKEN. Connect to https://api.example.invalid/v1 "
        "or 192.0.2.44.",
        source_name="SKILLS.md",
    )

    entities = static_entities(source)
    observed = {(entity.type, entity.value) for entity in entities}

    assert (EntityType.FILE, "package.json") in observed
    assert (EntityType.ENVIRONMENT_VARIABLE, "DEMO_TOKEN") in observed
    assert (EntityType.NETWORK_DESTINATION, "https://api.example.invalid/v1") in observed
    assert (EntityType.NETWORK_DESTINATION, "192.0.2.44") in observed
    assert all(entity.confidence == 1.0 for entity in entities)
    assert all(
        entity.trust is TrustLevel.EXTERNAL
        for entity in entities
        if entity.type is EntityType.NETWORK_DESTINATION
    )


def test_static_entities_keep_local_network_destinations_out_of_external_rules() -> None:
    source = SourceArtifact.from_text(
        "Connect to http://localhost:8080, 127.0.0.1, 10.0.0.7, or fe80::1.",
        source_name="local.md",
    )

    entities = static_entities(source)

    assert all(
        entity.trust is TrustLevel.UNKNOWN
        for entity in entities
        if entity.type is EntityType.NETWORK_DESTINATION
    )


def test_static_entities_use_clear_syntax_for_bare_domains_and_file_names() -> None:
    source = SourceArtifact.from_text(
        "Download from helper.example.org. Read package.json. Read yarn.lock. "
        "Use Next.js and React.FC. Keep yarn.lock available.",
        source_name="SKILLS.md",
    )

    entities = static_entities(source)
    observed = {(entity.type, entity.value) for entity in entities}

    assert (EntityType.NETWORK_DESTINATION, "helper.example.org") in observed
    assert (EntityType.FILE, "package.json") in observed
    assert (EntityType.FILE, "yarn.lock") in observed
    assert (EntityType.NETWORK_DESTINATION, "yarn.lock") not in observed
    assert (EntityType.NETWORK_DESTINATION, "React.FC") not in observed
    assert (EntityType.FILE, "Next.js") not in observed


def test_static_entities_accept_reviewed_domain_suffixes_only() -> None:
    source = SourceArtifact.from_text(
        "Use helper.example.de, helper.example.com, and helper.example.org. "
        "Do not classify yarn.lock or React.ChangeEvent as destinations.",
        source_name="domains.md",
    )

    destinations = {
        entity.value
        for entity in static_entities(source)
        if entity.type is EntityType.NETWORK_DESTINATION
    }

    assert {"helper.example.de", "helper.example.com", "helper.example.org"} <= destinations
    assert "yarn.lock" not in destinations
    assert "React.ChangeEvent" not in destinations
