"""Thermal R-JPEG parser.

Parses DJI / FLIR radiometric JPEGs to extract per-pixel temperature values
in Celsius. The DJI Thermal SDK integration is deferred until licensing is
resolved; a placeholder backend is provided for development.

Implementation lands in MVP-1 / MVP-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ThermalImage:
    """Parsed radiometric thermal image."""

    path: Path
    temperature_c: np.ndarray  # shape (H, W), dtype float32
    emissivity: float
    reflected_temp_c: float
    distance_m: float
    metadata: dict[str, Any]


def parse_thermal(image_path: str | Path, **calibration: Any) -> ThermalImage:
    """Parse an R-JPEG into a 2-D Celsius temperature array.

    Args:
        image_path: Path to an R-JPEG file.
        **calibration: Optional overrides for emissivity, reflected temperature,
            distance, humidity, atmospheric temperature.

    Returns:
        :class:`ThermalImage` with per-pixel temperatures in Celsius.

    Raises:
        NotImplementedError: Until DJI Thermal SDK / placeholder backend is wired.
    """
    raise NotImplementedError("Thermal parser implementation lands in MVP-1/MVP-3.")
