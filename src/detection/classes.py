"""Detection class definitions.

Single source of truth for class IDs and human-readable names used across
training, inference, reporting, and the dashboard. Kept in sync with
``config/model_config.yaml``.
"""

from __future__ import annotations

from enum import IntEnum


class DetectionClass(IntEnum):
    """Object detection class IDs.

    Placeholder roster - MVP-2 will revise based on dataset coverage.
    """

    POLE = 0
    INSULATOR = 1
    CONDUCTOR = 2
    DAMPER = 3
    SPACER = 4
    BIRD_NEST = 5
    DAMAGED_INSULATOR = 6
    MISSING_PART = 7


CLASS_NAMES: dict[int, str] = {c.value: c.name.lower() for c in DetectionClass}
