"""Hotspot detector for radiometric thermal images (MVP-3).

Identifies thermal anomalies (hotspots) in a per-pixel temperature map (°C) using
three strategies and a unified API:

* ``absolute``    — fixed temperature thresholds.
* ``statistical`` — outliers relative to the scene (median + k·std).
* ``hybrid``      — must satisfy BOTH an absolute floor and a statistical deviation.

Each strategy runs connected-component analysis on the candidate mask and reports a
:class:`Hotspot` per region (bbox, peak/mean temp, area, risk level, confidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from src.thermal.thresholds import ABSOLUTE_CRITICAL_MIN, ABSOLUTE_NORMAL_MAX

RiskLevel = Literal["low", "medium", "high", "critical"]
Method = Literal["absolute", "statistical", "hybrid"]

# Temperature (°C) assumed as the top of the meaningful range for confidence scaling.
_TEMP_CEILING = 150.0


@dataclass
class Hotspot:
    """A detected hotspot with bbox + temperature stats."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    peak_temp_c: float
    mean_temp_c: float
    area_pixels: int
    risk_level: RiskLevel
    confidence: float  # 0.0 - 1.0


@dataclass
class DetectionResult:
    """Output of hotspot detection for one image."""

    method: str
    hotspots: list[Hotspot] = field(default_factory=list)
    image_stats: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Shared helpers                                                               #
# --------------------------------------------------------------------------- #
def _image_stats(thermal: np.ndarray) -> dict:
    return {
        "min": float(thermal.min()),
        "max": float(thermal.max()),
        "mean": float(thermal.mean()),
        "std": float(thermal.std()),
        "median": float(np.median(thermal)),
    }


def _clamp01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _components_from_mask(
    thermal: np.ndarray,
    mask: np.ndarray,
    min_area: int,
    risk_fn,
    confidence_fn,
) -> list[Hotspot]:
    """Run connected-component analysis on ``mask`` and build a Hotspot per region.

    ``risk_fn(peak, mean) -> RiskLevel`` and ``confidence_fn(peak, mean) -> float``
    let each method decide risk/confidence from the region's measured temperatures.
    """
    mask_u8 = (mask.astype(np.uint8)) * 255
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    hotspots: list[Hotspot] = []
    for label in range(1, num):  # 0 is background
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue
        region = thermal[y:y + h, x:x + w]
        # Restrict stats to the masked pixels within the bbox (not the cool corners).
        region_mask = mask[y:y + h, x:x + w]
        vals = region[region_mask]
        if vals.size == 0:
            continue
        peak = float(vals.max())
        mean = float(vals.mean())
        hotspots.append(
            Hotspot(
                bbox=(int(x), int(y), int(w), int(h)),
                peak_temp_c=round(peak, 2),
                mean_temp_c=round(mean, 2),
                area_pixels=int(area),
                risk_level=risk_fn(peak, mean),
                confidence=round(confidence_fn(peak, mean), 3),
            )
        )
    # Strongest first.
    hotspots.sort(key=lambda hs: hs.peak_temp_c, reverse=True)
    return hotspots


# --------------------------------------------------------------------------- #
# Method 1: Absolute threshold                                                 #
# --------------------------------------------------------------------------- #
def detect_absolute_threshold(
    thermal: np.ndarray,
    low_threshold: float = ABSOLUTE_NORMAL_MAX,      # 60°C — endüstri standardı eşik
    medium_threshold: float = ABSOLUTE_CRITICAL_MIN,  # 80°C — endüstri standardı eşik
    high_threshold: float = 100.0,
    min_area: int = 50,
) -> DetectionResult:
    """Detect hotspots using absolute temperature thresholds.

    A pixel is a candidate if its temperature exceeds ``low_threshold``. Connected
    components yield regions; each region's peak temperature sets its risk level.
    """
    mask = thermal > low_threshold

    def risk_fn(peak: float, _mean: float) -> RiskLevel:
        if peak < medium_threshold:
            return "low"
        if peak < high_threshold:
            return "medium"
        if peak < high_threshold + 20:
            return "high"
        return "critical"

    def conf_fn(peak: float, _mean: float) -> float:
        denom = _TEMP_CEILING - low_threshold
        return _clamp01((peak - low_threshold) / denom) if denom > 0 else 0.0

    hotspots = _components_from_mask(thermal, mask, min_area, risk_fn, conf_fn)
    return DetectionResult(
        method="absolute",
        hotspots=hotspots,
        image_stats=_image_stats(thermal),
        parameters={
            "low_threshold": low_threshold,
            "medium_threshold": medium_threshold,
            "high_threshold": high_threshold,
            "min_area": min_area,
        },
    )


# --------------------------------------------------------------------------- #
# Method 2: Statistical / relative threshold                                   #
# --------------------------------------------------------------------------- #
def detect_statistical_threshold(
    thermal: np.ndarray,
    sigma_threshold: float = 3.0,
    min_temp_diff: float = 15.0,
    min_area: int = 50,
) -> DetectionResult:
    """Detect hotspots as statistical outliers relative to the scene.

    A pixel qualifies if it exceeds ``mean + sigma_threshold*std`` AND is at least
    ``min_temp_diff`` °C above the scene median (ambient).
    """
    ambient = float(np.median(thermal))
    std = float(np.std(thermal))
    threshold = max(ambient + sigma_threshold * std, ambient + min_temp_diff)
    mask = thermal > threshold

    def risk_fn(peak: float, _mean: float) -> RiskLevel:
        diff = peak - ambient
        if diff < 20:
            return "low"
        if diff < 40:
            return "medium"
        if diff < 60:
            return "high"
        return "critical"

    def conf_fn(peak: float, _mean: float) -> float:
        # How far past the threshold the peak sits, relative to its lift above ambient.
        span = peak - ambient
        return _clamp01((peak - threshold) / span) if span > 0 else 0.0

    hotspots = _components_from_mask(thermal, mask, min_area, risk_fn, conf_fn)
    return DetectionResult(
        method="statistical",
        hotspots=hotspots,
        image_stats=_image_stats(thermal),
        parameters={
            "sigma_threshold": sigma_threshold,
            "min_temp_diff": min_temp_diff,
            "min_area": min_area,
            "ambient": round(ambient, 2),
            "threshold": round(threshold, 2),
        },
    )


# --------------------------------------------------------------------------- #
# Method 3: Hybrid                                                             #
# --------------------------------------------------------------------------- #
def detect_hybrid(
    thermal: np.ndarray,
    absolute_min: float = 50.0,
    sigma_threshold: float = 2.5,
    min_temp_diff: float = 15.0,
    min_area: int = 50,
) -> DetectionResult:
    """Combine absolute and statistical thresholds.

    A pixel is a hotspot only if it is BOTH genuinely hot (> ``absolute_min``) AND a
    statistical outlier (> ``mean + sigma*std`` and >= ``min_temp_diff`` above ambient).
    """
    ambient = float(np.median(thermal))
    std = float(np.std(thermal))
    stat_threshold = max(ambient + sigma_threshold * std, ambient + min_temp_diff)

    abs_mask = thermal > absolute_min
    stat_mask = thermal > stat_threshold
    mask = abs_mask & stat_mask

    def risk_fn(peak: float, _mean: float) -> RiskLevel:
        # Blend absolute peak and lift-above-ambient; take the more severe verdict.
        diff = peak - ambient
        by_abs = ("critical" if peak >= 110 else "high" if peak >= 85
                  else "medium" if peak >= 65 else "low")
        by_diff = ("critical" if diff >= 60 else "high" if diff >= 40
                   else "medium" if diff >= 20 else "low")
        order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        names = ["low", "medium", "high", "critical"]
        return names[max(order[by_abs], order[by_diff])]  # type: ignore[return-value]

    def conf_fn(peak: float, _mean: float) -> float:
        # Average of absolute-scaled and statistical-scaled confidence.
        c_abs = (peak - absolute_min) / (_TEMP_CEILING - absolute_min)
        span = peak - ambient
        c_stat = (peak - stat_threshold) / span if span > 0 else 0.0
        return _clamp01(0.5 * c_abs + 0.5 * c_stat)

    hotspots = _components_from_mask(thermal, mask, min_area, risk_fn, conf_fn)
    return DetectionResult(
        method="hybrid",
        hotspots=hotspots,
        image_stats=_image_stats(thermal),
        parameters={
            "absolute_min": absolute_min,
            "sigma_threshold": sigma_threshold,
            "min_temp_diff": min_temp_diff,
            "min_area": min_area,
            "ambient": round(ambient, 2),
            "stat_threshold": round(stat_threshold, 2),
        },
    )


# --------------------------------------------------------------------------- #
# Unified API                                                                  #
# --------------------------------------------------------------------------- #
def detect_hotspots(
    thermal: np.ndarray,
    method: Method = "hybrid",
    **kwargs,
) -> DetectionResult:
    """Unified API. Routes to the chosen detection method."""
    if thermal.ndim != 2:
        raise ValueError(f"Expected a 2-D temperature map, got shape {thermal.shape}")
    if method == "absolute":
        return detect_absolute_threshold(thermal, **kwargs)
    if method == "statistical":
        return detect_statistical_threshold(thermal, **kwargs)
    if method == "hybrid":
        return detect_hybrid(thermal, **kwargs)
    raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------- #
# Evaluation helper                                                            #
# --------------------------------------------------------------------------- #
def bbox_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    """Intersection over Union for two [x, y, w, h] bboxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax2, ay2 = x1 + w1, y1 + h1
    bx2, by2 = x2 + w2, y2 + h2

    inter_x1, inter_y1 = max(x1, x2), max(y1, y2)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0
