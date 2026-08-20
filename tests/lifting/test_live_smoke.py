"""Optional real capability test for the explicit live Responses adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from nlir.lifting.live import CapabilityCheckResult, check_capability


def _smoke_config() -> Path | None:
    """Return the explicit config only when every real-call gate is set."""
    if os.environ.get("NLIR_LIVE_SMOKE") != "1":
        return None
    if not os.environ.get("NLIR_LIVE_API_KEY", "").strip():
        return None
    value = os.environ.get("NLIR_LIVE_SMOKE_CONFIG", "")
    return Path(value) if value else None


@pytest.mark.live_smoke
def test_live_capability_smoke_is_explicitly_enabled() -> None:
    """Call only the public harmless capability check after all gates pass."""
    config = _smoke_config()
    if config is None:
        pytest.skip("set NLIR_LIVE_SMOKE=1, NLIR_LIVE_API_KEY, and NLIR_LIVE_SMOKE_CONFIG")

    result = check_capability(config)

    assert isinstance(result, CapabilityCheckResult)
    assert result.available, [diagnostic.code for diagnostic in result.diagnostics]
