"""Public, offline CLI for deterministic source observation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from nlir.artifacts.extract import extract_annotations
from nlir.artifacts.loader import (
    LoadedArtifact,
    LoadFailure,
    PathDiagnostic,
    load_file,
    scan_directory,
    scan_loaded_artifact,
)
from nlir.ir import ArtifactRecord, LiftDiagnostic, hunt_records, lift_loaded_artifact
from nlir.rules.loader import load_rule
from nlir.rules.models import Rule

if TYPE_CHECKING:
    from nlir.benchmark import LoadedBenchmarkCase
    from nlir.ir import LiftRecordMetadata
    from nlir.lifting.models import SemanticLifter

app = typer.Typer(no_args_is_help=True, add_completion=False)
scan_app = typer.Typer(no_args_is_help=True, add_completion=False)
lift_app = typer.Typer(no_args_is_help=True, add_completion=False)
benchmark_app = typer.Typer(no_args_is_help=True, add_completion=False)
app.add_typer(scan_app, name="scan")
app.add_typer(lift_app, name="lift")
app.add_typer(benchmark_app, name="benchmark")


@scan_app.command("file")
def scan_file_command(path: Annotated[Path, typer.Argument(exists=False)]) -> None:
    """Scan one supported regular text file and print a JSON observation report."""
    try:
        loaded = load_file(path)
    except LoadFailure as error:
        _fail(error.diagnostic)
    typer.echo(_serialize_report((loaded,), ()))


@scan_app.command("directory")
def scan_directory_command(path: Annotated[Path, typer.Argument(exists=False)]) -> None:
    """Scan supported files below a directory and print a JSON observation report."""
    try:
        result = scan_directory(path)
    except LoadFailure as error:
        _fail(error.diagnostic)
    typer.echo(_serialize_report(result.artifacts, result.diagnostics))


@lift_app.command("live")
def lift_live_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to non-secret live TOML configuration"),
    ],
    path: Annotated[Path | None, typer.Argument(exists=False)] = None,
    check: Annotated[
        bool,
        typer.Option("--check", help="Check the live capability without lifting a source file"),
    ] = False,
    show: Annotated[
        bool,
        typer.Option("--show", help="Include accepted canonical IR in the JSON output"),
    ] = False,
    test_rule: Annotated[
        Path | None,
        typer.Option("--test-rule", help="Run one YAML rule on this lift only"),
    ] = None,
) -> None:
    """Lift one source through the explicit live path, or check that path."""
    from nlir.lifting.live import LiveResponsesLifter, check_capability

    if check:
        if path is not None:
            _fail(
                _lift_diagnostic(
                    "check_has_source", "The capability check does not accept a source path."
                )
            )
        if show or test_rule is not None:
            _fail(
                _lift_diagnostic(
                    "check_has_lift_options",
                    "The capability check cannot show IR or test a rule.",
                )
            )
        result = check_capability(config)
        payload = {
            "available": result.available,
            "diagnostics": [diagnostic.code for diagnostic in result.diagnostics],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        if not result.available:
            _fail(
                result.diagnostics[0]
                if result.diagnostics
                else _lift_diagnostic(
                    "capability_unavailable", "The live capability check failed."
                )
            )
        return
    if path is None:
        _fail(_lift_diagnostic("missing_source", "The live lift requires one source path."))
    rule = None
    if test_rule is not None:
        loaded_rule = load_rule(test_rule)
        if loaded_rule.rule is None:
            _fail(loaded_rule.diagnostics[0])
        rule = loaded_rule.rule
    try:
        loaded = load_file(path)
    except LoadFailure as error:
        _fail(error.diagnostic)
    lifter = LiveResponsesLifter.from_toml_file(config)
    metadata = lifter.lift_metadata()
    if metadata is None:
        diagnostic = lifter.setup_diagnostic or _lift_diagnostic(
            "invalid_live_config", "Live configuration is invalid."
        )
        _fail(diagnostic)
    records = lift_loaded_artifact(loaded, lifter=lifter, metadata=metadata)
    payload = _live_lift_report(records, metadata.model_dump(mode="json"))
    if show:
        payload["ir"] = _live_ir_report(records)
    if rule is not None:
        payload["rule_test"] = _live_rule_test(records, rule.id, rule)
    typer.echo(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    diagnostics = [
        diagnostic
        for record in records
        for attempt in record.canonical_attempts
        for diagnostic in attempt.diagnostics
    ]
    if diagnostics:
        _fail(diagnostics[0])


@app.command("web")
def web_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to non-secret live TOML configuration"),
    ],
    rules_directory: Annotated[
        Path,
        typer.Option("--rules-directory", help="Directory that contains local YAML rules"),
    ] = Path("rules"),
    host: Annotated[
        str,
        typer.Option("--host", help="Local browser server host"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", min=1, max=65_535, help="Local browser server port"),
    ] = 5000,
) -> None:
    """Start the local browser inspector for live prompt analysis."""
    from nlir.web.app import create_app

    try:
        flask_app = create_app(live_config=config, rules_directory=rules_directory)
    except ValueError as error:
        _fail(_lift_diagnostic("invalid_web_setup", str(error)))
    typer.echo(f"Open http://{host}:{port} in a browser. Press Ctrl+C to stop the server.")
    flask_app.run(host=host, port=port)


@benchmark_app.command("live")
def benchmark_live_command(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to non-secret live TOML configuration"),
    ],
    manifest: Annotated[
        Path,
        typer.Option("--manifest", help="Path to the benchmark manifest JSON"),
    ] = Path("benchmark/manifest.json"),
    rules_directory: Annotated[
        Path,
        typer.Option("--rules-directory", help="Directory that contains YAML rules"),
    ] = Path("rules"),
    family: Annotated[
        str | None,
        typer.Option("--family", help="Run only the named benchmark family"),
    ] = None,
) -> None:
    """Lift the full benchmark corpus live and report near-miss rule regressions."""
    from nlir.benchmark import BenchmarkManifestError, load_benchmark
    from nlir.lifting.live import LiveResponsesLifter

    try:
        corpus = load_benchmark(manifest)
    except BenchmarkManifestError as error:
        _fail(_lift_diagnostic("invalid_benchmark_manifest", str(error)))

    rules: list[Rule] = []
    for path in sorted(rules_directory.glob("*.yaml")):
        loaded_rule = load_rule(path)
        if loaded_rule.rule is None:
            _fail(loaded_rule.diagnostics[0])
        rules.append(loaded_rule.rule)

    lifter = LiveResponsesLifter.from_toml_file(config)
    metadata = lifter.lift_metadata()
    if metadata is None:
        diagnostic = lifter.setup_diagnostic or _lift_diagnostic(
            "invalid_live_config", "Live configuration is invalid."
        )
        _fail(diagnostic)

    cases = corpus.cases
    if family is not None:
        cases = tuple(item for item in cases if item.case.family == family)
        if not cases:
            _fail(_lift_diagnostic("unknown_family", f"No benchmark cases for family {family!r}."))

    report = _run_benchmark_live(cases, lifter=lifter, metadata=metadata, rules=tuple(rules))
    typer.echo(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _run_benchmark_live(
    cases: tuple[LoadedBenchmarkCase, ...],
    *,
    lifter: SemanticLifter,
    metadata: LiftRecordMetadata,
    rules: tuple[Rule, ...],
) -> dict[str, object]:
    """Lift every case live, hunt it with every rule, and group results by family."""
    case_results: dict[str, dict[str, object]] = {}
    for loaded_case in cases:
        records = lift_loaded_artifact(
            LoadedArtifact(artifact=loaded_case.artifact, relative_path=loaded_case.case.case_id),
            lifter=lifter,
            metadata=metadata,
        )
        incomplete = [
            {"ordinal": attempt.ordinal, "diagnostics": [d.code for d in attempt.diagnostics]}
            for record in records
            for attempt in record.canonical_attempts
            if attempt.canonical_fragment is None
        ]
        hits = sorted(
            {
                rule.id
                for rule in rules
                if any(result.status == "HIT" for result in hunt_records(records, rule).results)
            }
        )
        case_results[loaded_case.case.case_id] = {
            "family": loaded_case.case.family,
            "role": loaded_case.case.role,
            "modality": loaded_case.case.modality,
            "hits": hits,
            "incomplete_attempts": incomplete,
        }

    families: dict[str, dict[str, object]] = {}
    family_names = sorted({item["family"] for item in case_results.values()})
    for family_name in family_names:
        family_cases = {
            case_id: item
            for case_id, item in case_results.items()
            if item["family"] == family_name
        }
        risky_id = next(
            (case_id for case_id, item in family_cases.items() if item["role"] == "risky"), None
        )
        risky_hits = set(family_cases[risky_id]["hits"]) if risky_id else set()
        near_misses = [
            {
                "case_id": case_id,
                "modality": item["modality"],
                "hits": item["hits"],
                "regressions": sorted(set(item["hits"]) & risky_hits),
                "incomplete_attempts": item["incomplete_attempts"],
            }
            for case_id, item in sorted(family_cases.items())
            if item["role"] == "near_miss"
        ]
        families[family_name] = {
            "risky_case_id": risky_id,
            "risky_hits": sorted(risky_hits),
            "risky_incomplete_attempts": (
                family_cases[risky_id]["incomplete_attempts"] if risky_id else []
            ),
            "near_misses": near_misses,
        }

    return {
        "families": families,
        "summary": {
            "families_with_zero_coverage": sorted(
                name
                for name, fam in families.items()
                if fam["risky_case_id"] and not fam["risky_hits"]
            ),
            "families_with_regressions": sorted(
                name
                for name, fam in families.items()
                if any(near_miss["regressions"] for near_miss in fam["near_misses"])
            ),
            "total_regressions": sum(
                len(near_miss["regressions"])
                for fam in families.values()
                for near_miss in fam["near_misses"]
            ),
        },
    }


def _serialize_report(
    loaded_artifacts: tuple[LoadedArtifact, ...], diagnostics: tuple[PathDiagnostic, ...]
) -> str:
    """Build stable JSON from immutable source data without changing its text."""
    scanned = tuple(scan_loaded_artifact(item) for item in loaded_artifacts)
    artifacts = [item.loaded.artifact.model_dump(mode="json") for item in scanned]
    occurrences = [
        {"path": item.loaded.relative_path, "artifact_id": item.loaded.artifact.artifact_id}
        for item in scanned
    ]
    annotations = [
        annotation.model_dump(mode="json")
        for item in scanned
        for annotation in extract_annotations(item.loaded.artifact)
    ]
    virtual_children = [
        {
            "artifact": child.artifact.model_dump(mode="json"),
            "annotations": [annotation.model_dump(mode="json") for annotation in child.annotations],
        }
        for item in scanned
        for child in item.decoded.children
    ]
    decode_diagnostics = [
        diagnostic.model_dump(mode="json")
        for item in scanned
        for diagnostic in item.decoded.diagnostics
    ]
    report = {
        "artifacts": artifacts,
        "occurrences": occurrences,
        "annotations": annotations,
        "virtual_children": virtual_children,
        "decode_diagnostics": decode_diagnostics,
        "diagnostics": [asdict(diagnostic) for diagnostic in diagnostics],
    }
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _live_lift_report(
    records: tuple[ArtifactRecord, ...], metadata: dict[str, object]
) -> dict[str, object]:
    """Build a stable live report with identities and diagnostic codes only."""
    attempts: list[dict[str, object]] = []
    artifact_ids: list[str] = []
    unpack_diagnostics: list[dict[str, object]] = []
    for record in sorted(records, key=lambda item: item.source.artifact_id):
        artifact_ids.append(record.source.artifact_id)
        unpack_codes = [
            diagnostic.code
            for diagnostic in record.decode_diagnostics
            if diagnostic.code.startswith("unpack_")
        ]
        if unpack_codes:
            unpack_diagnostics.append(
                {
                    "artifact_id": record.source.artifact_id,
                    "diagnostics": unpack_codes,
                }
            )
        for attempt in record.canonical_attempts:
            attempts.append(
                {
                    "artifact_id": record.source.artifact_id,
                    "ordinal": attempt.ordinal,
                    "state": attempt.stage.value,
                    "diagnostics": [diagnostic.code for diagnostic in attempt.diagnostics],
                }
            )
    report: dict[str, object] = {
        "artifact_ids": artifact_ids,
        "attempts": sorted(
            attempts, key=lambda item: (str(item["artifact_id"]), int(item["ordinal"]))
        ),
        "metadata": metadata,
    }
    if unpack_diagnostics:
        report["unpack_diagnostics"] = unpack_diagnostics
    return report


def _live_ir_report(records: tuple[ArtifactRecord, ...]) -> list[dict[str, object]]:
    """Return accepted canonical fragments only after an explicit operator request."""
    return [
        {
            "artifact_id": record.source.artifact_id,
            "attempt_ordinal": attempt.ordinal,
            "canonical_fragment": attempt.canonical_fragment.model_dump(mode="json"),
        }
        for record in sorted(records, key=lambda item: item.source.artifact_id)
        for attempt in record.canonical_attempts
        if attempt.canonical_fragment is not None
    ]


def _live_rule_test(
    records: tuple[ArtifactRecord, ...], rule_id: str, rule: Rule
) -> dict[str, object]:
    """Return one binary rule result for each accepted attempt from this lift."""
    report = hunt_records(records, rule)
    results: list[dict[str, object]] = []
    for result in report.results:
        item: dict[str, object] = {
            "artifact_id": result.artifact_id,
            "attempt_ordinal": result.attempt_ordinal,
            "status": result.status,
        }
        if result.status == "HIT":
            item["hints"] = [hint.model_dump(mode="json") for hint in result.hints]
        results.append(item)
    return {
        "rule_id": rule_id,
        "results": results,
    }


def _lift_diagnostic(code: str, message: str) -> LiftDiagnostic:
    """Create a small CLI diagnostic for one local operation."""
    return LiftDiagnostic(code=code, message=message)


def _fail(diagnostic: object) -> None:
    """Print one typed local diagnostic and stop with an actionable status."""
    code = getattr(diagnostic, "code", "operation_failed")
    path = getattr(diagnostic, "path", "local")
    message = getattr(diagnostic, "message", "The operation failed.")
    typer.echo(f"error [{code}] {path}: {message}", err=True)
    raise typer.Exit(code=2)
