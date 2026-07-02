"""Tests for src.ingestion.exif_reader."""

from __future__ import annotations

import pytest

from src.ingestion.exif_reader import read_exif


def test_read_exif_not_implemented() -> None:
    """Placeholder - replaced when MVP-1 lands."""
    with pytest.raises(NotImplementedError):
        read_exif("does_not_exist.jpg")
