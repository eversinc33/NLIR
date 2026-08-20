"""Lifted IR contracts and the offline rule engine that reads them."""

from nlir.ir.models import (
    IR_FORMAT,
    ArtifactRecord,
    HuntReport,
    HuntResult,
    LiftDiagnostic,
    LiftedIR,
    LiftMetadata,
    LiftRecordMetadata,
    LiveLiftMetadata,
    SourceLocationHint,
)
from nlir.ir.service import (
    hunt_records,
    lift_loaded_artifact,
    resolve_input_source_span,
)

__all__ = [
    "IR_FORMAT",
    "ArtifactRecord",
    "HuntReport",
    "HuntResult",
    "LiftDiagnostic",
    "LiftMetadata",
    "LiftRecordMetadata",
    "LiftedIR",
    "LiveLiftMetadata",
    "SourceLocationHint",
    "hunt_records",
    "lift_loaded_artifact",
    "resolve_input_source_span",
]
