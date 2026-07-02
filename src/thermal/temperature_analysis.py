"""Temperature analysis helpers.

Statistical summaries and severity classification over thermal frames or
hotspot regions. Implementation lands in MVP-3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemperatureStats:
    """Aggregate temperature statistics over a region or full frame."""

    min_c: float
    max_c: float
    mean_c: float
    median_c: float
    std_c: float


def compute_stats(temperature_c: np.ndarray, mask: np.ndarray | None = None) -> TemperatureStats:
    """Compute per-pixel temperature statistics.

    Args:
        temperature_c: 2-D Celsius temperature array.
        mask: Optional boolean mask; statistics computed over ``True`` pixels.

    Returns:
        :class:`TemperatureStats`.

    Raises:
        NotImplementedError: Until MVP-3 implementation lands.
    """
    raise NotImplementedError("Temperature analysis implementation lands in MVP-3.")
