"""Strict, offline loading for the inert synthetic benchmark corpus.

The JSON manifest is the only control-plane input.  Fixture files are always
read as strict UTF-8 text, including files whose names end in ``.yaml``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from nlir.artifacts.models import SourceArtifact
from nlir.contracts.common import SourceSpan, StrictFrozenModel


class BenchmarkManifestError(ValueError):
    """A manifest or fixture failed a corpus safety or integrity check."""


Family = Literal[
    "credential_external_transfer",
    "package_install",
    "instruction_override_privileged_action",
    "decoded_instruction_action",
    "configuration_change_persistence",
    "openclaw_memory_persistence",
    "transparency_suppression",
    "log_deletion",
    "parseltongue_binary_spacing",
    "parseltongue_reversed_octets",
    "parseltongue_custom_bijection",
]
Role = Literal["risky", "near_miss"]
Location = Literal[
    "markdown_body",
    "yaml_metadata",
    "json_string",
    "markdown_reference",
    "encoded_virtual_child",
]
Language = Literal["en", "de", "fr", "es"]
NearMissModality = Literal[
    "negation",
    "descriptive",
    "policy_detection",
    "quoted_example",
    "hypothetical",
    "attacker_description",
]
ModalityLabel = Literal[
    "imperative",
    "negation",
    "descriptive",
    "policy_detection",
    "quoted_example",
    "hypothetical",
    "attacker_description",
]
AnchorName = Literal["source", "action", "sink", "modality"]
CaseId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,80}$")]
RuleId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{2,80}$")]

_UNSAFE_MARKERS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"(?m)^#!"),
    re.compile(r"\brm\s+-rf\b"),
)
_URL = re.compile(r"https?://([^/\s]+)", re.IGNORECASE)


class EvidenceAnchor(StrictFrozenModel):
    """A named, unique literal that becomes derived source evidence."""

    name: AnchorName
    literal: Annotated[str, Field(min_length=1, max_length=256)]


class SemanticFactAssertion(StrictFrozenModel):
    """A future semantic assertion tied to one or more named evidence anchors."""

    fact: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_:-]{2,80}$")]
    anchors: Annotated[tuple[AnchorName, ...], Field(min_length=1)]


class FutureRuleAssertion(StrictFrozenModel):
    """A future rule finding assertion tied to named source evidence."""

    rule_id: RuleId
    anchors: Annotated[tuple[AnchorName, ...], Field(min_length=1)]


class BenchmarkCase(StrictFrozenModel):
    """One immutable, inert benchmark case described by the JSON manifest."""

    case_id: CaseId
    fixture_path: Annotated[str, Field(min_length=1, max_length=512)]
    family: Family
    role: Role
    location: Location
    language: Language
    modality: ModalityLabel
    paired_with: CaseId | None
    evidence_anchors: Annotated[tuple[EvidenceAnchor, ...], Field(min_length=4, max_length=4)]
    expected_facts: Annotated[tuple[SemanticFactAssertion, ...], Field(min_length=1)]
    forbidden_facts: Annotated[tuple[SemanticFactAssertion, ...], Field(min_length=1)]
    expected_rules: tuple[FutureRuleAssertion, ...] = ()
    forbidden_rules: tuple[FutureRuleAssertion, ...] = ()

    @model_validator(mode="after")
    def validate_case_contract(self) -> BenchmarkCase:
        path = PurePosixPath(self.fixture_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or "\\" in self.fixture_path
        ):
            raise ValueError("fixture_path must stay below benchmark fixture root")

        names = tuple(anchor.name for anchor in self.evidence_anchors)
        literals = tuple(anchor.literal for anchor in self.evidence_anchors)
        if set(names) != {"source", "action", "sink", "modality"} or len(set(names)) != 4:
            raise ValueError(
                "evidence anchors must name source, action, sink, and modality exactly once"
            )
        if len(set(literals)) != len(literals):
            raise ValueError("evidence anchor literals must be unique")

        known_anchors = set(names)
        declarations = (
            *self.expected_facts,
            *self.forbidden_facts,
            *self.expected_rules,
            *self.forbidden_rules,
        )
        if any(set(declaration.anchors) - known_anchors for declaration in declarations):
            raise ValueError("future assertions must reference declared evidence anchors")
        if self.role == "risky":
            if self.paired_with is not None or self.modality != "imperative":
                raise ValueError("risky cases require imperative modality and no paired_with")
        else:
            if self.paired_with is None or self.modality == "imperative":
                raise ValueError("near misses require a risky paired_with and near-miss modality")
        return self


class BenchmarkManifest(StrictFrozenModel):
    """Frozen v1.0 corpus catalog; fixtures are deliberately not embedded."""

    corpus_version: Literal["1.0"]
    cases: Annotated[tuple[BenchmarkCase, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_pairing(self) -> BenchmarkManifest:
        ids = tuple(case.case_id for case in self.cases)
        if len(set(ids)) != len(ids):
            raise ValueError("case IDs must be unique")
        by_id = {case.case_id: case for case in self.cases}
        for case in self.cases:
            if case.role == "near_miss":
                pair = by_id.get(case.paired_with or "")
                if pair is None or pair.role != "risky" or pair.family != case.family:
                    raise ValueError(
                        f"near miss {case.case_id} must pair with a risky seed in its family"
                    )
        return self


@dataclass(frozen=True)
class LoadedBenchmarkCase:
    """A manifest case with its preserved text and derived evidence spans."""

    case: BenchmarkCase
    artifact: SourceArtifact
    anchor_spans: dict[AnchorName, SourceSpan]


@dataclass(frozen=True)
class BenchmarkCorpus:
    """A fully validated, text-only benchmark manifest and its fixtures."""

    manifest: BenchmarkManifest
    cases: tuple[LoadedBenchmarkCase, ...]


def load_benchmark(manifest_path: Path) -> BenchmarkCorpus:
    """Load JSON manifest plus inert strict-UTF-8 fixture text without executing it."""
    try:
        raw_manifest = _freeze_json_arrays(json.loads(manifest_path.read_text(encoding="utf-8")))
        manifest = BenchmarkManifest.model_validate(raw_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as error:
        raise BenchmarkManifestError(f"invalid benchmark manifest: {error}") from error

    fixture_root = (manifest_path.parent / "fixtures").resolve()
    loaded_cases = tuple(_load_case(case, fixture_root) for case in manifest.cases)
    return BenchmarkCorpus(manifest=manifest, cases=loaded_cases)


def _load_case(case: BenchmarkCase, fixture_root: Path) -> LoadedBenchmarkCase:
    fixture_path = (fixture_root / case.fixture_path).resolve()
    try:
        fixture_path.relative_to(fixture_root)
    except ValueError as error:
        raise BenchmarkManifestError(
            f"case {case.case_id}: fixture path must stay below benchmark fixture root"
        ) from error
    try:
        with fixture_path.open("r", encoding="utf-8", errors="strict", newline="") as fixture_file:
            text = fixture_file.read()
    except (OSError, UnicodeDecodeError) as error:
        raise BenchmarkManifestError(
            f"case {case.case_id}: cannot read fixture: {error}"
        ) from error

    _assert_safe_fixture(case.case_id, text)
    artifact = SourceArtifact.from_text(text, source_name=case.fixture_path)
    anchors: dict[AnchorName, SourceSpan] = {}
    for anchor in case.evidence_anchors:
        occurrences = text.count(anchor.literal)
        if occurrences != 1:
            raise BenchmarkManifestError(
                f"case {case.case_id}: anchor {anchor.name!r} must occur exactly once, "
                f"found {occurrences}"
            )
        start = text.index(anchor.literal)
        anchors[anchor.name] = SourceSpan(
            artifact_id=artifact.artifact_id,
            start=start,
            end=start + len(anchor.literal),
        )
    return LoadedBenchmarkCase(case=case, artifact=artifact, anchor_spans=anchors)


def _assert_safe_fixture(case_id: str, text: str) -> None:
    for marker in _UNSAFE_MARKERS:
        if marker.search(text):
            raise BenchmarkManifestError(f"case {case_id}: unsafe fixture content marker")
    for match in _URL.finditer(text):
        host = match.group(1).split("@")[-1].split(":")[0].lower()
        if not host.endswith(".invalid"):
            raise BenchmarkManifestError(
                f"case {case_id}: unsafe destination {host!r}; use .invalid"
            )


def _freeze_json_arrays(value: object) -> object:
    """Translate JSON's sole array representation into immutable contract tuples."""
    if isinstance(value, list):
        return tuple(_freeze_json_arrays(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_json_arrays(item) for key, item in value.items()}
    return value
