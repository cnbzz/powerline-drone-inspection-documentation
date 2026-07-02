"""Excel report generator (openpyxl).

Writes a structured Excel workbook with summary and per-fault detail sheets.
Implementation lands in MVP-5.
"""

from __future__ import annotations

from pathlib import Path


def generate_excel(faults: list, output_path: str | Path) -> Path:
    """Write a fault list to an ``.xlsx`` workbook.

    Args:
        faults: Iterable of :class:`~src.fault.fault_record.FaultRecord`.
        output_path: Destination ``.xlsx`` path.

    Returns:
        The written file path.

    Raises:
        NotImplementedError: Until MVP-5 implementation lands.
    """
    raise NotImplementedError("Excel report generator implementation lands in MVP-5.")
