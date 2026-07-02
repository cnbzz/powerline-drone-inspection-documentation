"""RGB / thermal image pairing.

Matches RGB and thermal frames captured by dual-sensor drones (e.g. DJI M3T,
H20T) based on filename conventions and capture timestamps.

Implementation lands in MVP-1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImagePair:
    """Paired RGB and thermal capture from the same moment."""

    rgb_path: Path
    thermal_path: Path
    delta_seconds: float


def pair_images(rgb_dir: str | Path, thermal_dir: str | Path) -> list[ImagePair]:
    """Pair RGB and thermal files from two directories.

    Args:
        rgb_dir: Directory containing RGB images.
        thermal_dir: Directory containing thermal (R-JPEG) images.

    Returns:
        List of :class:`ImagePair`, one per matched capture.

    Raises:
        NotImplementedError: Until MVP-1 implementation lands.
    """
    raise NotImplementedError("RGB/thermal pairing implementation lands in MVP-1.")
