"""Public library API for lifting and hunting natural-language behavior."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from nlir.artifacts.loader import LoadFailure, load_file
from nlir.ir import (
    ArtifactRecord,
    HuntReport,
    LiftedIR,
    LiftRecordMetadata,
    hunt_records,
    lift_loaded_artifact,
)
from nlir.lifting.models import SemanticLifter
from nlir.rules.loader import load_rule
from nlir.rules.models import Rule, RuleDiagnostic


class RuleLoadError(ValueError):
    """Stop a public rule-read operation with its typed loader diagnostic."""

    def __init__(self, diagnostic: RuleDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class NLIR:
    """Library-first interface for reading rules, lifting files, and hunting IR.

    Lifting returns a serializable :class:`~nlir.ir.LiftedIR` value.  Keeping that
    value is the caller's responsibility; this library never writes it anywhere.
    """

    LiftedIR = LiftedIR

    def __init__(
        self,
        *,
        lifter: SemanticLifter | None = None,
        metadata: LiftRecordMetadata | None = None,
    ) -> None:
        if (lifter is None) != (metadata is None):
            raise ValueError("lifter and metadata must be configured together")
        self._lifter = lifter
        self._metadata = metadata
        self._closed = False

    @classmethod
    def from_live_config(cls, config: str | Path) -> NLIR:
        """Create a live library instance from one explicit non-secret TOML file."""
        from nlir.lifting.live import LiveResponsesLifter

        lifter = LiveResponsesLifter.from_toml_file(config)
        metadata = lifter.lift_metadata()
        if metadata is None:
            raise ValueError("The live configuration is invalid.")
        return cls(lifter=lifter, metadata=metadata)

    def __enter__(self) -> NLIR:
        """Open this library session."""
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        """Close this library session."""
        self.close()

    def close(self) -> None:
        """Close this library session. It holds no lifted IR of its own."""
        self._closed = True

    def read_rule(self, path: str | Path) -> Rule:
        """Read one strict YAML rule or raise its typed loader error."""
        self._require_open()
        loaded = load_rule(Path(path))
        if loaded.rule is None:
            raise RuleLoadError(loaded.diagnostics[0])
        return loaded.rule

    def read_rule_dir(self, directory: str | Path) -> tuple[Rule, ...]:
        """Read every YAML rule in one directory in stable filename order."""
        self._require_open()
        rule_directory = Path(directory)
        if not rule_directory.is_dir():
            raise ValueError("The rule directory is not available.")
        return tuple(self.read_rule(path) for path in sorted(rule_directory.glob("*.yaml")))

    def lift_file(self, path: str | Path) -> LiftedIR:
        """Lift one file and return its complete IR for the caller to keep."""
        self._require_open()
        if self._lifter is None or self._metadata is None:
            raise RuntimeError("A configured lifter and metadata are required to lift files.")
        try:
            loaded = load_file(Path(path))
        except LoadFailure as error:
            raise ValueError(error.diagnostic.message) from error
        return LiftedIR(
            records=lift_loaded_artifact(
                loaded,
                lifter=self._lifter,
                metadata=self._metadata,
            )
        )

    def run_rule(self, rule: Rule, lifted_ir: LiftedIR | Iterable[LiftedIR]) -> HuntReport:
        """Run one rule over one lift result or over several kept lift results."""
        self._require_open()
        results = (lifted_ir,) if isinstance(lifted_ir, LiftedIR) else tuple(lifted_ir)
        records: tuple[ArtifactRecord, ...] = tuple(
            record for result in results for record in result.records
        )
        return hunt_records(records, rule)

    def _require_open(self) -> None:
        """Stop calls after the context manager closes."""
        if self._closed:
            raise RuntimeError("This NLIR session is closed.")
