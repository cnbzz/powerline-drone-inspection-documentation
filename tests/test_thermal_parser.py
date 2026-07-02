"""Tests for src.ingestion.thermal_parser."""

from __future__ import annotations

import pytest

from src.ingestion.thermal_parser import parse_thermal


def test_parse_thermal_not_implemented() -> None:
    """Placeholder - replaced when MVP-1/MVP-3 lands."""
    with pytest.raises(NotImplementedError):
        parse_thermal("does_not_exist.jpg")
