"""EXIF metadata reader.

Extracts GPS, timestamp, altitude, camera model and other relevant EXIF fields
from drone-captured RGB images.

Implementation lands in MVP-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExifMetadata:
    """Structured EXIF metadata of a single image."""

    path: Path
    captured_at: datetime | None
    latitude: float | None
    longitude: float | None
    altitude_m: float | None
    camera_make: str | None
    camera_model: str | None
    raw: dict[str, Any]


def read_exif(image_path: str | Path) -> ExifMetadata:
    """Read EXIF metadata from a single image file.

    Args:
        image_path: Path to an RGB image (JPEG/TIFF) with EXIF data.

    Returns:
        Parsed :class:`ExifMetadata`.

    Raises:
        FileNotFoundError: If ``image_path`` does not exist.
        NotImplementedError: Until MVP-1 implementation lands.
    """
    raise NotImplementedError("EXIF reader implementation lands in MVP-1.")
