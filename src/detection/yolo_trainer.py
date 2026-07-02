"""YOLO trainer wrapper.

Thin wrapper around :mod:`ultralytics` for training YOLOv8/v11 models with
project-level configuration. Implementation lands in MVP-2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_yolo(
    data_yaml: str | Path,
    base_model: str = "yolov8n.pt",
    **overrides: Any,
) -> Path:
    """Train a YOLO model on the project dataset.

    Args:
        data_yaml: Path to YOLO ``data.yaml``.
        base_model: Pretrained checkpoint to start from.
        **overrides: Hyperparameter overrides (epochs, batch, lr, ...).

    Returns:
        Path to the best checkpoint produced by ultralytics.

    Raises:
        NotImplementedError: Until MVP-2 implementation lands.
    """
    raise NotImplementedError("YOLO trainer implementation lands in MVP-2.")
