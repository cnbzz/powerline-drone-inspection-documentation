"""PDF report generator (ReportLab).

Renders per-line / per-flight fault inspection reports as PDF, using
templates under ``src/reporting/templates``. Implementation lands in MVP-5.
"""

from __future__ import annotations

from pathlib import Path


def generate_pdf(
    faults: list,
    output_path: str | Path,
    *,
    title: str = "Powerline Inspection Report",
) -> Path:
    """Render a fault list to a PDF report.

    Args:
        faults: Iterable of :class:`~src.fault.fault_record.FaultRecord`.
        output_path: Destination ``.pdf`` path.
        title: Document title.

    Returns:
        The written file path.

    Raises:
        NotImplementedError: Until MVP-5 implementation lands.
    """
    raise NotImplementedError("PDF report generator implementation lands in MVP-5.")
