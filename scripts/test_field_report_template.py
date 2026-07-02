"""BEDAŞ resmi saha raporu şablonu doğrulama testi.

Üretir:
  (a) model teşhisi (polimer + rust) + termal OLAN bir arıza → tam rapor
  (b) termal-only / model teşhisi olmayan bir arıza → graceful bölümler

Doğrular: sayfa sayısı > 1 (PageBreak), exception yok, PDF içinde emoji yok.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.fault.database import FaultDatabase  # noqa: E402
from src.dashboard.utils.pdf.field_report import generate_field_report  # noqa: E402

DB = 'outputs/db/faults.db'
OUT = _ROOT / 'outputs' / 'reports'

# Faults sayfasındaki DataFrame kolonlarını yansıtan minimal Series kurucu.
RISK_TR = {'critical': 'Kritik', 'high': 'Yüksek', 'medium': 'Orta', 'low': 'Düşük'}
STATUS_TR = {'new': 'Yeni', 'reviewed': 'İncelendi', 'resolved': 'Çözüldü',
             'false_positive': 'Yanlış Alarm'}


def _series_from_record(r, *, inject_thermal=None) -> pd.Series:
    from src.defect_classifier.inference import YOLO_TO_DEFECT_EQUIPMENT  # noqa: F401
    equipment_label = r.equipment_class or '—'
    peak = r.thermal.peak_temp_c if r.thermal else None
    delta = r.thermal.temperature_delta_c if r.thermal else None
    thermal_status = None
    if r.thermal:
        from src.thermal.thresholds import assess_thermal
        thermal_status = assess_thermal(r.thermal.peak_temp_c,
                                        r.thermal.temperature_delta_c).final_status.value
    if inject_thermal is not None:
        peak, delta, thermal_status = inject_thermal

    return pd.Series({
        'fault_id': r.fault_id,
        'fault_id_short': r.fault_id[:8],
        'timestamp': r.timestamp,
        'flight_label': r.inspection_flight_id or '—',
        'fault_type_label': r.fault_type.value,
        'equipment_class': r.equipment_class,
        'equipment_label': equipment_label,
        'risk_level': r.risk_level.value,
        'risk_score': r.risk_score,
        'status_tr': STATUS_TR.get(r.status.value, r.status.value),
        'detection_confidence': r.detection_confidence,
        'latitude': r.gps.latitude if r.gps else None,
        'longitude': r.gps.longitude if r.gps else None,
        'peak_temp': peak,
        'delta_temp': delta,
        'thermal_status': thermal_status,
        'assigned_to_name': '—',
    })


def _page_count(pdf_path: Path) -> int:
    try:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    except Exception:
        # Fallback: count "/Type /Page" occurrences
        data = pdf_path.read_bytes()
        return data.count(b'/Type /Page') - data.count(b'/Type /Pages')


def _extract_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(str(pdf_path)).pages)
    except Exception:
        return ''


def _emoji_chars(text: str) -> list:
    found = []
    for ch in text:
        cp = ord(ch)
        if (0x1F000 <= cp <= 0x1FAFF) or (0x2600 <= cp <= 0x27BF) or cp in (0x2B50, 0x2B55):
            found.append(ch)
    return sorted(set(found))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    db = FaultDatabase(DB)
    rows = {r.fault_id[:8]: r for r in db.list_all(limit=10000)}

    # (a) polimer + rust + (enjekte) termal kritik → tam rapor
    rec_a = rows['c75a66c5']
    series_a = _series_from_record(rec_a, inject_thermal=(85.0, 35.0, 'critical'))
    path_a = OUT / 'TEST_template_a_full_c75a66c5.pdf'
    bytes_a = generate_field_report(series_a, output_path=path_a)

    # (b) termal-only, model teşhisi yok → graceful
    rec_b = rows['a695e81e']
    series_b = _series_from_record(rec_b)
    path_b = OUT / 'TEST_template_b_thermal_only_a695e81e.pdf'
    bytes_b = generate_field_report(series_b, output_path=path_b)

    print('=' * 64)
    all_ok = True
    for label, path, raw in [('(a) FULL  ', path_a, bytes_a),
                             ('(b) THERM ', path_b, bytes_b)]:
        pages = _page_count(path)
        text = _extract_text(path)
        emojis = _emoji_chars(text)
        valid_pdf = raw[:4] == b'%PDF'
        ok = valid_pdf and pages > 1 and not emojis
        all_ok = all_ok and ok
        print(f'{label} {path.name}')
        print(f'         size={len(raw)//1024}KB  pages={pages}  '
              f'valid_pdf={valid_pdf}  emojis={emojis if emojis else "NONE"}')
        # Spot-check banned model names absent
        for banned in ('YOLOv8', 'EfficientNet', 'Stage 1', 'Stage 2', 'Grad-CAM', 'AI Teşhisi'):
            if banned in text:
                print(f'         !! BANNED TERM FOUND: {banned}')
                all_ok = False

    print('=' * 64)
    print('Paths:')
    print('  ', path_a)
    print('  ', path_b)
    print('=' * 64)
    print('ALL TEMPLATE CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
    if not all_ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
