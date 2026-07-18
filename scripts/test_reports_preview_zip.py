"""Standalone verification for Reports A1: PDF bytes, base64 preview HTML, and a
multi-select ZIP (3 reports). Mirrors 4_Reports.load_all_faults row construction
so it exercises the SAME generate_field_report path the page uses.
"""
from __future__ import annotations

import base64
import io
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.dashboard.app import DB_PATH
from src.fault.database import FaultDatabase
from src.thermal.thresholds import assess_thermal
from src.dashboard.utils.pdf.field_report import generate_field_report

CLASS_NAMES_TR = {"insulator_glass": "Cam İzolatör", "insulator_polymer": "Polimer İzolatör"}
FAULT_TYPE_LABELS = {"thermal_hotspot": "Termal Anomali", "visual_defect": "Görsel Hasar"}


def _row(r) -> dict:
    """Same keys 4_Reports.load_all_faults produces (generate_field_report input)."""
    thermal_status = (
        assess_thermal(r.thermal.peak_temp_c, r.thermal.temperature_delta_c)
        .final_status.value if r.thermal else None
    )
    return {
        "fault_id": r.fault_id, "fault_id_short": r.fault_id[:8],
        "timestamp": r.timestamp,
        "flight_label": r.inspection_flight_id or "—",
        "fault_type_label": FAULT_TYPE_LABELS.get(r.fault_type.value, r.fault_type.value),
        "equipment_class": r.equipment_class,
        "equipment_label": CLASS_NAMES_TR.get(r.equipment_class, r.equipment_class or "—"),
        "risk_level": r.risk_level.value, "risk_score": r.risk_score,
        "status_tr": r.status.value, "detection_confidence": r.detection_confidence,
        "latitude": r.gps.latitude if r.gps else None,
        "longitude": r.gps.longitude if r.gps else None,
        "peak_temp": r.thermal.peak_temp_c if r.thermal else None,
        "delta_temp": r.thermal.temperature_delta_c if r.thermal else None,
        "thermal_status": thermal_status,
    }


def _pdf_preview_html(pdf_bytes: bytes) -> str:
    b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="780"></iframe>'


def main() -> None:
    db = FaultDatabase(str(DB_PATH))
    records = db.list_all(limit=10000)
    print(f"DB faults: {len(records)}")
    assert len(records) >= 3, "need >=3 faults to test ZIP"

    rows = [pd.Series(_row(r)) for r in records[:3]]
    fids = [row["fault_id_short"] for row in rows]
    print(f"Testing fids: {fids}")

    # 1) PDF bytes per fault
    pdfs = {}
    for row in rows:
        pdf = generate_field_report(row)
        fid = row["fault_id_short"]
        ok = isinstance(pdf, (bytes, bytearray)) and pdf[:4] == b"%PDF"
        print(f"  [{fid}] PDF bytes={len(pdf):>6}  valid_header={ok}")
        assert ok, f"invalid PDF for {fid}"
        pdfs[fid] = pdf

    # 2) base64 preview HTML valid + round-trips
    html = _pdf_preview_html(pdfs[fids[0]])
    assert "data:application/pdf;base64," in html and html.startswith("<iframe")
    b64 = html.split("base64,")[1].split('"')[0]
    assert base64.b64decode(b64) == pdfs[fids[0]], "base64 round-trip mismatch"
    print(f"[2] preview HTML OK (iframe + valid base64 data-URI, len={len(html)})")

    # 3) ZIP of the 3 selected reports
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fid in fids:
            zf.writestr(f"HHATS_Saha_Raporu_{fid}.pdf", pdfs[fid])
    buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        assert len(names) == 3, f"expected 3 PDFs, got {len(names)}"
        for n in names:
            assert zf.read(n)[:4] == b"%PDF", f"corrupt PDF in zip: {n}"
    print(f"[3] ZIP OK: {len(names)} PDFs, all valid -> {names}")

    print("=" * 60)
    print("ALL STANDALONE CHECKS PASSED")


if __name__ == "__main__":
    main()
