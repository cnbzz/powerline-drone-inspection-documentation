"""Visualization helpers.

Bounding-box overlays, severity colour maps, and folium marker builders.
Implementation grows alongside MVP-2..MVP-5.
"""

from __future__ import annotations

SEVERITY_COLORS: dict[str, str] = {
    "low": "#2ecc71",
    "medium": "#f1c40f",
    "high": "#e67e22",
    "critical": "#e74c3c",
}


def severity_color(severity: str) -> str:
    """Map a severity band to a hex colour.

    Args:
        severity: One of ``low``, ``medium``, ``high``, ``critical``.

    Returns:
        Hex colour string (e.g. ``"#e74c3c"``); ``"#7f8c8d"`` for unknown.
    """
    return SEVERITY_COLORS.get(severity.lower(), "#7f8c8d")
