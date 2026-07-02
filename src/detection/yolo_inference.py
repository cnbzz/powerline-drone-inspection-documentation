"""YOLO inference wrapper.

Runs detection on RGB images and returns structured results.
Implementation lands in MVP-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Detection:
    """A single bounding-box detection."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


def run_inference(
    model_path: str | Path,
    image_paths: list[str | Path],
    *,
    conf: float = 0.25,
    iou: float = 0.45,
) -> dict[Path, list[Detection]]:
    """Run YOLO inference over a batch of images.

    Args:
        model_path: Path to a trained ``.pt`` checkpoint.
        image_paths: Image files to run inference on.
        conf: Confidence threshold.
        iou: NMS IoU threshold.

    Returns:
        Mapping from image path to list of :class:`Detection`.

    Raises:
        NotImplementedError: Until MVP-2 implementation lands.
    """
    raise NotImplementedError("YOLO inference implementation lands in MVP-2.")
