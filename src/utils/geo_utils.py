"""Geo helpers.

Coordinate transforms, haversine distances, and helpers for projecting
image-frame detections to world coordinates given GPS + altitude + camera
intrinsics.
"""

from __future__ import annotations

from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_M: float = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in meters.

    Args:
        lat1: First point latitude in degrees.
        lon1: First point longitude in degrees.
        lat2: Second point latitude in degrees.
        lon2: Second point longitude in degrees.

    Returns:
        Distance in meters.
    """
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * asin(sqrt(a))
