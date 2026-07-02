"""Tests for src.fault.risk_scorer."""

from __future__ import annotations

import pytest

from src.fault.risk_scorer import RiskInput, score_fault


def test_score_fault_not_implemented() -> None:
    """Placeholder - replaced when MVP-4 lands."""
    inputs = RiskInput(
        visual_class="damaged_insulator",
        visual_confidence=0.9,
        delta_above_ambient_c=25.0,
        location_criticality=0.5,
        history_factor=0.2,
    )
    with pytest.raises(NotImplementedError):
        score_fault(inputs)
