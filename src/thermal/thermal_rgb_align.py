"""RGB / thermal frame alignment.

Estimates a homography between paired RGB and thermal frames so that
detection bounding boxes from RGB can be projected onto the thermal frame
(and vice-versa). Implementation lands in MVP-3.
"""

from __future__ import annotations

import numpy as np


def align_thermal_to_rgb(
    rgb: np.ndarray,
    thermal: np.ndarray,
    *,
    method: str = "ORB",
) -> np.ndarray:
    """Return a 3x3 homography that maps thermal pixels onto the RGB frame.

    Args:
        rgb: HxWx3 uint8 RGB image.
        thermal: HxW float32 temperature image (or its uint8 visualisation).
        method: ``"ORB"`` | ``"SIFT"`` | ``"manual"``.

    Returns:
        3x3 float64 homography matrix.

    Raises:
        NotImplementedError: Until MVP-3 implementation lands.
    """
    raise NotImplementedError("Thermal-RGB alignment implementation lands in MVP-3.")
