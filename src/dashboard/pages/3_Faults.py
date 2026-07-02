"""Arıza Yönetimi sayfası - MVP-5.3.

Filtrelenebilir arıza listesi + RGB/termal görüntü inceleme + status güncelleme.
Veri kaynağı: ``outputs/db/faults.db`` (uydurma veri yok). Görüntü dosyaları
``outputs/fault_files/fault_<short_id>/`` klasörlerinden okunur.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime  # noqa: E402
from typing import Optional  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Arıza Yönetimi - HHATS", page_icon="🚨", layout="wide")

from src.dashboard.app import (  # noqa: E402
    DB_PATH,
    get_current_theme,
    load_css,
    render_sidebar,
)
from src.dashboard.utils.excel_export import export_faults_to_excel  # noqa: E402
from src.fault.database import FaultDatabase  # noqa: E402
from src.fault.fault_record import FaultStatus  # noqa: E402
from src.thermal.thresholds import (  # noqa: E402
    STATUS_COLORS,
    STATUS_EMOJI,
    STATUS_LABELS_TR,
    ThermalStatus,
    assess_thermal,
)

# Stage 2 defect classifier (opsiyonel — model/paket yoksa UI yine çalışır)
try:
    import base64  # noqa: E402
    import cv2  # noqa: E402

    from src.defect_classifier.inference import (  # noqa: E402
        predict_defect,
        YOLO_TO_DEFECT_EQUIPMENT,
        CLASS_LABELS_TR,
    )
    STAGE2_AVAILABLE = True
except ImportError:
    STAGE2_AVAILABLE = False

load_css()

from src.auth.session import require_login  # noqa: E402

require_login()  # giriş kapısı — giriş yoksa login formu + st.stop()
render_sidebar()

# --------------------------------------------------------------------------- #
# Türkçe etiket sözlükleri (enum value → görünen ad)                          #
# NOT: FaultStatus enum'unda "yanlış alarm" → value "false_positive" (false_alarm değil)
# --------------------------------------------------------------------------- #
RISK_TR = {"critical": "Kritik", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}
STATUS_TR = {
    "new": "Yeni",
    "reviewed": "İncelendi",
    "resolved": "Çözüldü",
    "false_positive": "Yanlış Alarm",
}
FAULT_TYPE_LABELS = {
    "thermal_hotspot": "Termal Anomali",
    "visual_defect": "Görsel Hasar",
    "equipment_detected": "Normal Ekipman",
    "combined": "Çoklu Arıza",
    "vegetation_encroachment": "Bitki Örtüsü",
    "unknown": "Belirsiz",
}
CLASS_NAMES_TR = {
    "insulator_polymer": "Polimer İzolatör",
    "insulator_glass": "Cam İzolatör",
    "insulator_porcelain": "Porselen İzolatör",
    "insulator_generic": "İzolatör",
    "tower": "Direk",
    "spacer": "Spacer",
    "damper": "Damper",
    "tower_id_plate": "Direk Plakası",
}
FLIGHT_LABELS = {
    "flight_2026_05_01_kocaeli": "Kocaeli-Gebze (01.05.2026)",
    "flight_2026_05_08_sakarya": "Sakarya-Adapazarı (08.05.2026)",
    "flight_2026_05_15_bursa": "Bursa-İnegöl (15.05.2026)",
    "flight_2026_05_22_istanbul": "İstanbul-Çatalca (22.05.2026)",
}

OUTPUTS_DIR = _ROOT / "outputs" / "fault_files"

# Türkçe-duyarlı arama normalizasyonu. Gerekçe: "Cam İzolatör"deki İ (U+0130)
# ASCII 'i'ye case-fold OLMAZ → kullanıcı "izolat" yazınca pandas str.contains(case=False)
# hiçbir şey bulmaz. Tüm Türkçe karakterleri ASCII'ye indirip lower() yapıyoruz.
_TR_MAP = str.maketrans({
    "İ": "i", "I": "i", "ı": "i", "Ş": "s", "ş": "s", "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u", "Ö": "o", "ö": "o", "Ç": "c", "ç": "c",
})


def _tr_normalize(text: str) -> str:
    """Türkçe karakterleri ASCII'ye indir + küçült (aksana/dotted-İ duyarsız arama)."""
    if not text:
        return ""
    return text.translate(_TR_MAP).lower()


@st.cache_data(ttl=30)
def load_all_faults() -> pd.DataFrame:
    """DB'den tüm fault records — 30sn cache."""
    db = FaultDatabase(str(DB_PATH))
    rows = []
    for r in db.list_all(limit=10000):
        equipment_label = CLASS_NAMES_TR.get(r.equipment_class, r.equipment_class or "—")
        fault_type_label = FAULT_TYPE_LABELS.get(r.fault_type.value, r.fault_type.value)
        notes = r.notes or ""
        rows.append({
            "fault_id": r.fault_id,
            "fault_id_short": r.fault_id[:8],
            "search_blob": _tr_normalize(
                f"{r.fault_id[:8]} {equipment_label} {fault_type_label} {notes}"
            ),
            "timestamp": r.timestamp,
            "flight_id": r.inspection_flight_id or "—",
            "flight_label": FLIGHT_LABELS.get(r.inspection_flight_id, r.inspection_flight_id or "—"),
            "fault_type": r.fault_type.value,
            "fault_type_label": fault_type_label,
            "equipment_class": r.equipment_class or "—",
            "equipment_label": equipment_label,
            "risk_level": r.risk_level.value,
            "risk_level_tr": RISK_TR.get(r.risk_level.value, r.risk_level.value),
            "risk_score": r.risk_score,
            "status": r.status.value,
            "status_tr": STATUS_TR.get(r.status.value, r.status.value),
            "detection_confidence": r.detection_confidence,
            "latitude": r.gps.latitude if r.gps else None,
            "longitude": r.gps.longitude if r.gps else None,
            "peak_temp": r.thermal.peak_temp_c if r.thermal else None,
            "delta_temp": r.thermal.temperature_delta_c if r.thermal else None,
            # Sabit endüstri eşikleriyle termal durum (normal/warning/critical) — None=termal veri yok
            "thermal_status": (
                assess_thermal(r.thermal.peak_temp_c,
                               r.thermal.temperature_delta_c).final_status.value
                if r.thermal else None
            ),
            "notes": notes,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=60)
def load_personnel_list() -> pd.DataFrame:
    """Aktif personel listesi - 60sn cache."""
    db = FaultDatabase(str(DB_PATH))
    rows = db.list_personnel(active_only=True)
    return pd.DataFrame([
        {"id": p.id, "name": p.name, "role": p.role, "team": p.team}
        for p in rows
    ])


@st.cache_data(ttl=10)
def load_fault_notes(fault_id: str) -> list[dict]:
    """Bir arızanın notları - 10sn cache (sık güncellenir)."""
    db = FaultDatabase(str(DB_PATH))
    notes = db.list_notes(fault_id)
    return [
        {
            "id": n.id, "type": n.note_type.value, "content": n.content,
            "author": n.author, "created_at": n.created_at,
        }
        for n in notes
    ]


@st.cache_data(ttl=10)
def load_fault_assignment(fault_id: str) -> dict:
    """Atama bilgisi - 10sn cache."""
    db = FaultDatabase(str(DB_PATH))
    return db.get_fault_assignment(fault_id) or {}


def update_fault_status(fault_id: str, new_status: str) -> bool:
    """DB'de fault status güncelle ve cache'i temizle."""
    try:
        db = FaultDatabase(str(DB_PATH))
        ok = db.update_status(fault_id, FaultStatus(new_status), reviewer="dashboard_user")
        if not ok:
            st.error("Kayıt bulunamadı, status güncellenemedi.")
            return False
        st.cache_data.clear()  # load_all_faults + sidebar stats cache'lerini tazele
        return True
    except Exception as e:  # noqa: BLE001
        st.error(f"Status güncellenemedi: {e}")
        return False


def render_metric_card(label: str, value: str, color: str = None) -> None:
    theme = get_current_theme()
    if color is None:
        color = theme['accent_primary']
    st.markdown(f"""
    <div style="background:{theme['bg_tertiary']};
                border:1px solid {theme['border_light']};
                border-left:4px solid {color};
                border-radius:8px; padding:0.8rem 1rem; text-align:center;">
        <div style="color:{theme['text_secondary']}; font-size:0.75rem; font-weight:600;
                    text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
        <div style="color:{theme['text_primary']}; font-size:1.6rem; font-weight:700;
                    margin-top:0.2rem;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def _collapse_html(html: str) -> str:
    """st.markdown render fix: baştaki girintiyi sil, boş satırları at.

    Streamlit st.markdown'ı Python-Markdown ile işler; 4+ boşlukla başlayan
    satır 'indented code block' sayılıp escape edilir (HTML çıplak metin görünür).
    Her satırı strip edip boş satırları atınca tek bitişik HTML bloğu kalır →
    HTML olarak render edilir. unsafe_allow_html=True yine gerekli (mevcut).
    """
    return "\n".join(s for s in (ln.strip() for ln in html.splitlines()) if s)


def render_faults_table_html(disp: pd.DataFrame, theme: dict) -> str:
    """Faults tablosunu tema uyumlu pure-HTML tablo olarak render et.

    st.dataframe (Glide Canvas Grid) CSS'i kabul etmiyor; HTML tablo = pure DOM,
    tema dict değişince renkler GARANTİ uygulanır.
    ``disp`` Türkçe kolon adlarıyla gelir: ID, Tarih, Risk, Skor, Tip, Ekipman, Uçuş, Durum.
    """
    risk_styles = {
        'Kritik': f'background-color:{theme["danger"]}; color:white; font-weight:600; padding:4px 10px; border-radius:4px;',
        'Yüksek': f'background-color:{theme["orange"]}; color:white; font-weight:600; padding:4px 10px; border-radius:4px;',
        'Orta':   f'background-color:{theme["warning"]}; color:{theme["text_primary"]}; font-weight:600; padding:4px 10px; border-radius:4px;',
        'Düşük':  f'background-color:{theme["success"]}; color:white; font-weight:600; padding:4px 10px; border-radius:4px;',
    }
    status_styles = {
        'Yeni':         f'background-color:{theme["accent_primary"]}; color:white; padding:4px 10px; border-radius:4px;',
        'İncelendi':    f'background-color:{theme["warning"]}; color:{theme["text_primary"]}; padding:4px 10px; border-radius:4px;',
        'Çözüldü':      f'background-color:{theme["success"]}; color:white; padding:4px 10px; border-radius:4px;',
        'Yanlış Alarm': f'background-color:{theme["text_secondary"]}; color:white; padding:4px 10px; border-radius:4px;',
    }
    thermal_styles = {
        'Normal': f'background-color:{STATUS_COLORS[ThermalStatus.NORMAL]}; color:white; font-weight:600; padding:4px 10px; border-radius:4px;',
        'Dikkat': f'background-color:{STATUS_COLORS[ThermalStatus.WARNING]}; color:#1c1c1e; font-weight:600; padding:4px 10px; border-radius:4px;',
        'Kritik': f'background-color:{STATUS_COLORS[ThermalStatus.CRITICAL]}; color:white; font-weight:600; padding:4px 10px; border-radius:4px;',
    }

    def _cell(fid: str, content: str, extra: str = "") -> str:
        """Tüm hücre tıklanabilir olsun diye içeriği ?fault_detail linkiyle sar.

        NOT: Streamlit st.markdown inline `onclick`/`onmouseover` JS'ini sanitize
        ederek SİLER → satır tıklaması <a href> ile yapılır (KPI kartlarıyla aynı,
        kanıtlanmış desen). Hover efekti CSS `.fault-row:hover` ile (app.py).
        """
        return (
            f'<td style="padding:0;">'
            f'<a href="?fault_detail={fid}" target="_self" '
            f'style="display:block; padding:10px 12px; text-decoration:none; '
            f'color:{theme["text_primary"]}; {extra}">{content}</a></td>'
        )

    rows_html = ""
    for _, row in disp.iterrows():
        fid = row['_fault_id_full']
        risk_html = f'<span style="{risk_styles.get(row["Risk"], "")}">{row["Risk"]}</span>'
        status_html = f'<span style="{status_styles.get(row["Durum"], "")}">{row["Durum"]}</span>'
        # Termal badge ("🟡 Dikkat" → durum metni "Dikkat" stil için ayrıştırılır)
        thermal_label = row.get('Termal', '—')
        if thermal_label and thermal_label != '—' and ' ' in thermal_label:
            tstatus = thermal_label.split(' ', 1)[1]
            thermal_html = f'<span style="{thermal_styles.get(tstatus, "")}">{thermal_label}</span>'
        else:
            thermal_html = '<span style="color:#999;">—</span>'
        rows_html += f"""
        <tr class="fault-row" style="border-bottom:1px solid {theme['border_light']}; cursor:pointer;">
            {_cell(fid, row['ID'], "font-family:monospace;")}
            {_cell(fid, row['Tarih'])}
            {_cell(fid, risk_html)}
            {_cell(fid, thermal_html)}
            {_cell(fid, f"{row['Skor']:.1f}", "text-align:right;")}
            {_cell(fid, row['Tip'])}
            {_cell(fid, row['Ekipman'])}
            {_cell(fid, row['Uçuş'])}
            {_cell(fid, status_html)}
        </tr>
        """

    th = (f"padding:12px; text-align:left; color:{theme['text_primary']}; font-weight:600;")
    th_r = (f"padding:12px; text-align:right; color:{theme['text_primary']}; font-weight:600;")
    table_html = f"""
    <div style="background:{theme['bg_tertiary']}; border-radius:12px; overflow-x:auto;
                border:1px solid {theme['border_light']}; margin:1rem 0;">
        <table style="width:100%; border-collapse:collapse;
                      font-family:-apple-system, 'Segoe UI', sans-serif; font-size:0.9rem;">
            <thead>
                <tr style="background:{theme['bg_primary']};
                           border-bottom:2px solid {theme['border_medium']};">
                    <th style="{th}">ID</th>
                    <th style="{th}">Tarih</th>
                    <th style="{th}">Risk</th>
                    <th style="{th}">Termal</th>
                    <th style="{th_r}">Skor</th>
                    <th style="{th}">Tip</th>
                    <th style="{th}">Ekipman</th>
                    <th style="{th}">Uçuş</th>
                    <th style="{th}">Durum</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
    """
    return _collapse_html(table_html)


@st.cache_data(ttl=300, show_spinner=False)
def _generate_field_report_pdf(fault_id_short: str):
    """PDF saha raporu (cache'li — arıza başına bir kez üretilir, Stage 2 dahil).

    download_button verisi önceden gerekli olduğu için detay paneli render
    edilirken üretilir; cache sayesinde her rerun'da yeniden üretilmez.
    """
    try:
        from src.dashboard.utils.pdf.field_report import generate_field_report
    except ImportError:
        return None
    df = load_all_faults()
    matches = df[df['fault_id_short'] == fault_id_short]
    if matches.empty:
        return None
    try:
        return generate_field_report(matches.iloc[0])
    except Exception:  # noqa: BLE001
        return None


def get_thermal_recommendation(status: ThermalStatus) -> str:
    """Termal duruma göre Türkçe öneri metni."""
    if status == ThermalStatus.CRITICAL:
        return ("⚠️ ACİL: Saha ekibi 24-48 saat içinde müdahale etmelidir. "
                "Aşırı ısınma kalıcı hasara yol açabilir.")
    elif status == ThermalStatus.WARNING:
        return ("📋 DİKKAT: Planlı bakım programına eklenmelidir. "
                "Önümüzdeki 2 hafta içinde kontrol edilmelidir.")
    return "✅ NORMAL: Ekipman güvenli sıcaklık aralığında çalışıyor."


def render_risk_explanation_panel(fault: pd.Series, theme: dict) -> None:
    """Risk skorunun TAM açıklaması — şeffaflık paneli (Stage 7.1A).

    Mevcut risk hesaplamasını DEĞİŞTİRMEZ; kayıtlı breakdown'ı (record.risk_factors)
    insan-okur bileşen bar'larına çevirir. Toplam, gösterilen risk_score ile tutarlıdır.
    """
    fault_id = fault.get('fault_id')
    if not fault_id:
        return
    try:
        from src.fault.risk_scorer import explain_risk_score
        record = FaultDatabase(str(DB_PATH)).get(fault_id)
        if record is None:
            return
        breakdown = explain_risk_score(record)
    except Exception:  # noqa: BLE001
        return
    if not breakdown.components:
        return

    level_color = {
        'critical': theme['danger'], 'high': theme['orange'],
        'medium': theme['warning'], 'low': theme['success'],
    }.get(breakdown.risk_level, theme['text_secondary'])
    level_label_tr = {
        'critical': 'KRİTİK', 'high': 'YÜKSEK', 'medium': 'ORTA', 'low': 'DÜŞÜK',
    }.get(breakdown.risk_level, str(breakdown.risk_level).upper())
    threshold = {'critical': 75, 'high': 50, 'medium': 25, 'low': 0}.get(breakdown.risk_level, 0)

    # Bileşen çubukları (bar genişliği = ham skor 0-100; katkı puan olarak yazılır)
    component_bars = ""
    for comp in sorted(breakdown.components, key=lambda c: -c.contribution):
        bar_width = max(0, min(100, int(round(comp.raw_score))))
        if bar_width > 75:
            bar_color = theme['danger']
        elif bar_width > 50:
            bar_color = theme['orange']
        elif bar_width > 25:
            bar_color = theme['warning']
        else:
            bar_color = theme['success']
        component_bars += f"""
        <div style="margin:10px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="color:{theme['text_primary']}; font-weight:500;">{comp.name}</span>
                <span style="color:{theme['text_secondary']}; font-size:0.85rem;">
                    +{comp.contribution:.1f} puan · ham {comp.raw_score:.0f}/100 · ağırlık %{comp.weight*100:.0f}
                </span>
            </div>
            <div style="background:{theme['border_light']}; height:8px; border-radius:4px; overflow:hidden;">
                <div style="background:{bar_color}; height:100%; width:{bar_width}%;"></div>
            </div>
            <div style="color:{theme['text_secondary']}; font-size:0.78rem; margin-top:3px; font-style:italic;">
                {comp.explanation}
            </div>
        </div>
        """

    dominant = breakdown.get_dominant_component()
    dominant_text = ""
    if dominant:
        dominant_text = (
            f"Bu kayıt {level_label_tr} kategorisinde çünkü <b>{dominant.name}</b> en büyük "
            f"katkıyı yapıyor (+{dominant.contribution:.1f} puan): {dominant.explanation}"
        )

    st.markdown(_collapse_html(f"""
    <div style="background:{theme['bg_tertiary']}; padding:1.25rem 1.5rem;
                border-radius:12px; border:1px solid {theme['border_light']};
                border-left:4px solid {level_color}; margin:1.5rem 0;">
        <h4 style="margin:0 0 12px 0; color:{theme['text_primary']}; font-size:1.05rem; font-weight:600;">
            🧮 Bu Neden {level_label_tr}?
        </h4>
        <div style="background:{level_color}26; padding:0.75rem 1rem; border-radius:8px; margin-bottom:1rem;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <div style="font-size:1.5rem; font-weight:700; color:{level_color};">
                        {breakdown.total_score:.1f} / 100
                    </div>
                    <div style="color:{theme['text_secondary']}; font-size:0.85rem;">Toplam Risk Skoru</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.2rem; font-weight:700; color:{level_color};">{level_label_tr}</div>
                    <div style="color:{theme['text_secondary']}; font-size:0.78rem;">Eşik: ≥{threshold}</div>
                </div>
            </div>
        </div>
        <div style="margin-bottom:1rem;">
            <strong style="color:{theme['text_primary']};">Risk Bileşenleri (aktif olanlar):</strong>
            {component_bars}
        </div>
        <div style="background:{theme['bg_primary']}; padding:0.75rem 1rem; border-radius:8px;
                    border-left:3px solid {level_color};">
            <div style="color:{theme['text_primary']}; font-size:0.9rem;">
                <strong>SONUÇ:</strong> {dominant_text}
            </div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_ai_diagnosis_panel(fault: pd.Series, theme: dict) -> None:
    """AI Teşhisi paneli — Stage 2 defect classification + Grad-CAM.

    fault Series'inden `equipment_class` (YOLO sınıfı) ve `fault_id_short` kullanılır;
    crop görüntüsü outputs/fault_files/fault_<short>/01_rgb_original.jpg.
    """
    if not STAGE2_AVAILABLE:
        return

    yolo_class = fault.get('equipment_class', '')
    defect_equipment = YOLO_TO_DEFECT_EQUIPMENT.get(yolo_class)

    # Bu ekipman tipi için Stage 2 sınıflandırıcı yoksa bilgi ver
    if defect_equipment is None:
        st.markdown(_collapse_html(f"""
        <div style="background:{theme['bg_tertiary']}; padding:1rem 1.25rem;
                    border-radius:12px; border-left:4px solid {theme['text_secondary']};
                    margin:1.5rem 0;">
            <h4 style="margin:0 0 8px 0; color:{theme['text_primary']};">🤖 AI Teşhisi (Stage 2)</h4>
            <p style="margin:0; color:{theme['text_secondary']}; font-size:0.9rem;">
                Bu ekipman tipi için Stage 2 sınıflandırıcı henüz mevcut değil
                (sadece tespit yapıldı). Desteklenen: cam izolatör, polimer izolatör.
            </p>
        </div>
        """), unsafe_allow_html=True)
        return

    # Crop görüntüsü (orijinal RGB)
    fault_dir = OUTPUTS_DIR / f"fault_{fault['fault_id_short']}"
    image_path = fault_dir / "01_rgb_original.jpg"
    if not image_path.exists():
        st.markdown(_collapse_html(f"""
        <div style="background:{theme['bg_tertiary']}; padding:1rem 1.25rem;
                    border-radius:12px; border-left:4px solid {theme['text_secondary']};
                    margin:1.5rem 0;">
            <h4 style="margin:0 0 8px 0; color:{theme['text_primary']};">🤖 AI Teşhisi (Stage 2)</h4>
            <p style="margin:0; color:{theme['text_secondary']}; font-size:0.9rem;">
                Görüntü dosyası bulunamadı; AI teşhisi için crop görüntüsü gerekli
                (gerçek drone verisi bekleniyor).
            </p>
        </div>
        """), unsafe_allow_html=True)
        return

    # Stage 2 prediction
    try:
        result = predict_defect(image_path, yolo_class)
    except Exception as e:  # noqa: BLE001
        st.warning(f"AI Teşhisi sırasında hata: {e}")
        return
    if result is None:
        return

    accent_color = theme['danger'] if result['is_defective'] else theme['success']
    status_text = '⚠️ DEFEKT TESPİT EDİLDİ' if result['is_defective'] else '✅ NORMAL'

    # Olasılık çubukları
    prob_bars = ""
    for cls, prob in sorted(result['all_probabilities'].items(), key=lambda x: -x[1]):
        cls_tr = CLASS_LABELS_TR.get(cls, cls)
        bar_width = int(prob * 100)
        bar_color = accent_color if cls == result['predicted_class'] else theme['text_secondary']
        prob_bars += f"""
        <div style="margin:6px 0;">
            <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:3px;">
                <span style="color:{theme['text_primary']}; font-weight:500;">{cls_tr}</span>
                <span style="color:{theme['text_secondary']};">{prob*100:.1f}%</span>
            </div>
            <div style="background:{theme['border_light']}; height:6px; border-radius:3px; overflow:hidden;">
                <div style="background:{bar_color}; height:100%; width:{bar_width}%;"></div>
            </div>
        </div>
        """

    # Grad-CAM base64
    gradcam_html = ""
    if result.get('gradcam_overlay') is not None:
        ok, buffer = cv2.imencode('.jpg', result['gradcam_overlay'])
        if ok:
            b64 = base64.b64encode(buffer).decode('utf-8')
            gradcam_html = f"""
            <div style="margin-top:1rem;">
                <strong style="color:{theme['text_primary']};">🔥 Grad-CAM Görsel Kanıt:</strong>
                <p style="color:{theme['text_secondary']}; font-size:0.85rem; margin:6px 0;">
                    Modelin karar verirken baktığı bölgeler (kırmızı = en yüksek dikkat):
                </p>
                <img src="data:image/jpeg;base64,{b64}" style="max-width:100%; border-radius:8px; border:1px solid {theme['border_light']};">
            </div>
            """

    st.markdown(_collapse_html(f"""
    <div style="background:{theme['bg_tertiary']}; padding:1.25rem 1.5rem;
                border-radius:12px; border:1px solid {theme['border_light']};
                border-left:4px solid {accent_color}; margin:1.5rem 0;">
        <h4 style="margin:0 0 12px 0; color:{theme['text_primary']}; font-size:1.05rem; font-weight:600;">
            🤖 AI Teşhisi (Stage 2)
        </h4>
        <div style="background:{accent_color}15; padding:0.75rem 1rem; border-radius:8px; margin-bottom:1rem;">
            <div style="font-size:1.1rem; font-weight:700; color:{accent_color}; margin-bottom:4px;">
                {status_text}
            </div>
            <div style="color:{theme['text_primary']}; font-size:0.95rem;">
                Teşhis: <strong>{result['predicted_class_tr']}</strong>
            </div>
            <div style="color:{theme['text_secondary']}; font-size:0.85rem; margin-top:2px;">
                Güven: %{result['confidence']*100:.1f} | Model: EfficientNetV2-S | Ekipman: {defect_equipment}
            </div>
        </div>
        <div style="margin-bottom:1rem;">
            <strong style="color:{theme['text_primary']};">Tüm Olasılıklar:</strong>
            {prob_bars}
        </div>
        {gradcam_html}
    </div>
    """), unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _cached_defect_prediction(fault_id_short: str, yolo_class: str):
    """Forward-only Stage 2 tahmini (Grad-CAM YOK → hafif), arıza başına cache'li.

    XAI yorum paneli yalnız olasılık sayılarına ihtiyaç duyar; AI Teşhisi kutusunun
    (6.2) ayrı predict_with_gradcam çağrısını DEĞİŞTİRMEDEN tekrar koşmamak için
    burada yalnızca engine.predict() (forward) kullanılır.
    """
    if not STAGE2_AVAILABLE:
        return None
    try:
        from src.defect_classifier.inference import get_inference_engine
    except ImportError:
        return None
    equipment = YOLO_TO_DEFECT_EQUIPMENT.get(yolo_class)
    if equipment is None:
        return None
    img_path = OUTPUTS_DIR / f"fault_{fault_id_short}" / "01_rgb_original.jpg"
    if not img_path.exists():
        return None
    try:
        from PIL import Image
        with open(img_path, 'rb') as f:
            pil = Image.open(f).convert('RGB')
        return get_inference_engine().predict(pil, equipment)
    except Exception:  # noqa: BLE001
        return None


def render_model_explainability_panel(fault: pd.Series, theme: dict) -> None:
    """Stage 2 model kararı XAI yorum katmanı (Aşama 7.1B).

    Mevcut "AI Teşhisi" (6.2) kutusunun ALTINA, genişletilebilir bölüm. Olasılık
    çubuklarını TEKRARLAMAZ → kararın NE KADAR GÜVENİLİR olduğunu yorumlar.
    """
    if not STAGE2_AVAILABLE:
        return

    yolo_class = fault.get('equipment_class', '')
    defect_equipment = YOLO_TO_DEFECT_EQUIPMENT.get(yolo_class)
    fault_id_short = fault.get('fault_id_short', '')
    image_path = OUTPUTS_DIR / f"fault_{fault_id_short}" / "01_rgb_original.jpg"

    # Graceful: model teşhisi olmayan arıza (termal-only / desteklenmeyen ekipman)
    if defect_equipment is None or not image_path.exists():
        with st.expander("🔬 Model Karar Analizi (XAI)", expanded=False):
            st.markdown(_collapse_html(f"""
            <div style="color:{theme['text_secondary']}; font-size:0.9rem; padding:0.25rem 0;">
                Bu arıza için model teşhisi mevcut değil
                (yalnızca termal/desteklenmeyen ekipman tipi).
            </div>
            """), unsafe_allow_html=True)
        return

    prediction = _cached_defect_prediction(fault_id_short, yolo_class)
    if not prediction:
        return

    try:
        from src.defect_classifier.explainability import build_model_explanation
        me = build_model_explanation(prediction, defect_equipment)
    except Exception:  # noqa: BLE001
        return

    BEDAS_BLUE = '#003a8c'
    ACCENT_GREEN = '#009a3a'
    conf_color = {'high': ACCENT_GREEN, 'medium': '#ff9500', 'low': '#ff3b30'}[me.confidence_tier]
    conf_label = {'high': 'YÜKSEK GÜVEN', 'medium': 'ORTA GÜVEN',
                  'low': 'DÜŞÜK GÜVEN'}[me.confidence_tier]
    margin_color = {'clear': ACCENT_GREEN, 'moderate': '#ff9500',
                    'borderline': '#ff3b30'}[me.margin_tier]
    margin_label = {'clear': 'NET KARAR', 'moderate': 'ORTA',
                    'borderline': 'SINIR KARAR / BELİRSİZ'}[me.margin_tier]

    f1 = me.model_f1
    if f1 is None:
        f1_text = "Bu ekipman sınıfı için model başarı verisi bulunamadı."
    else:
        qual = ("çok yüksek" if f1 >= 0.95 else "yüksek" if f1 >= 0.90 else "orta")
        f1_text = (f"Bu ekipman sınıfı için modelin doğrulama başarısı: "
                   f"<b>F1 = {f1:.2f}</b> ({qual}).")

    # Karar marjı açıklaması (Türkçe)
    margin_expl = (f"1. ve 2. tahmin arası fark: <b>%{me.margin_pct:.0f}</b> → "
                   f"{'net karar' if me.margin_tier == 'clear' else 'orta düzey' if me.margin_tier == 'moderate' else 'sınır karar / belirsiz'}")

    # Belirsizlik / domain uyarıları
    caveats_html = ""
    if me.caveats:
        items = "".join(
            f'<li style="margin:4px 0; color:{theme["text_primary"]};">{c}</li>'
            for c in me.caveats
        )
        caveats_html = f"""
        <div style="margin-top:12px; background:#fff8e1; border-left:4px solid #ff9500;
                    padding:10px 14px; border-radius:6px;">
            <strong style="color:#b26a00;">⚠️ Belirsizlik / Domain Uyarıları:</strong>
            <ul style="margin:6px 0 0 0; padding-left:18px; font-size:0.86rem;">{items}</ul>
        </div>
        """

    verdict_color = (ACCENT_GREEN if me.overall_verdict.startswith('YÜKSEK')
                     else '#ff3b30' if me.overall_verdict.startswith('DÜŞÜK')
                     else '#ff9500')

    with st.expander("🔬 Model Karar Analizi (XAI)", expanded=False):
        st.markdown(_collapse_html(f"""
        <div style="font-family:-apple-system,'Segoe UI',sans-serif;">
            <div style="color:{BEDAS_BLUE}; font-weight:700; font-size:1rem; margin-bottom:10px;">
                EfficientNetV2-S kararının denetlenebilir analizi
            </div>

            <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                <div style="flex:1; min-width:160px; background:{theme['bg_primary']};
                            border-radius:8px; padding:10px 14px; border-left:4px solid {conf_color};">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
                                color:{theme['text_secondary']};">Karar Güveni (Top-1)</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{conf_color};">%{me.confidence_pct:.1f}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:{conf_color};">{conf_label}</div>
                </div>
                <div style="flex:1; min-width:160px; background:{theme['bg_primary']};
                            border-radius:8px; padding:10px 14px; border-left:4px solid {margin_color};">
                    <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
                                color:{theme['text_secondary']};">Karar Marjı (Top1−Top2)</div>
                    <div style="font-size:1.3rem; font-weight:700; color:{margin_color};">%{me.margin_pct:.0f}</div>
                    <div style="font-size:0.8rem; font-weight:600; color:{margin_color};">{margin_label}</div>
                </div>
            </div>

            <div style="font-size:0.9rem; color:{theme['text_primary']}; margin-bottom:8px;">{margin_expl}</div>

            <div style="background:{theme['bg_primary']}; border-radius:8px; padding:10px 14px;
                        margin-bottom:10px; border-left:4px solid {BEDAS_BLUE};">
                <strong style="color:{BEDAS_BLUE};">📊 Model Performans Bağlamı:</strong>
                <div style="font-size:0.88rem; color:{theme['text_primary']}; margin-top:4px;">{f1_text}</div>
            </div>

            <div style="background:{theme['bg_primary']}; border-radius:8px; padding:10px 14px;
                        margin-bottom:10px; border-left:4px solid {ACCENT_GREEN};">
                <strong style="color:{ACCENT_GREEN};">🔥 Görsel Odak Yorumu:</strong>
                <div style="font-size:0.86rem; color:{theme['text_secondary']}; margin-top:4px;">
                    Model kararını yukarıdaki Grad-CAM ısı haritasında vurgulanan bölgeye dayandırdı;
                    kırmızı alanlar modelin en çok dikkat ettiği yerlerdir.
                </div>
            </div>

            {caveats_html}

            <div style="margin-top:12px; background:{verdict_color}1A; border:1px solid {verdict_color};
                        border-radius:8px; padding:10px 14px;">
                <strong style="color:{verdict_color};">GENEL GÜVENİLİRLİK:</strong>
                <span style="color:{theme['text_primary']}; font-size:0.9rem;"> {me.overall_verdict}</span>
            </div>
        </div>
        """), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Kestirimci Bakım paneli (Aşama 7.4-UI) — bozulma tahmini + gelecek risk eğrisi #
# src/predictive/ modülünü ÇAĞIRIR (mantığını değiştirmez). Sentetik veri →     #
# akademik dürüstlük: disclaimer zorunlu.                                       #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=300, show_spinner=False)
def _cached_degradation_forecast(fault_id_short: str, risk_score: float,
                                 peak_temp, delta_temp, confidence,
                                 start_iso: Optional[str]):
    """Tek arıza için bozulma tahmini (cache'li). Model yok/eksik veri → None."""
    try:
        from datetime import date as _date

        from src.predictive.inference import predict_degradation

        # Tek-snapshot geçmiş: mevcut risk + (varsa) termal/güven. Eksikler
        # inference içinde risk'ten türetilir.
        obs = {"risk": float(risk_score)}
        if peak_temp is not None and not pd.isna(peak_temp):
            obs["peak_temp"] = float(peak_temp)
        if delta_temp is not None and not pd.isna(delta_temp):
            obs["delta_temp"] = float(delta_temp)
        if confidence is not None and not pd.isna(confidence):
            obs["confidence"] = float(confidence)

        start = None
        if start_iso:
            try:
                start = _date.fromisoformat(start_iso[:10])
            except Exception:
                start = None

        forecast = predict_degradation([obs], start_date=start)
        return forecast.to_dict() if forecast is not None else None
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=300, show_spinner=False)
def _build_degradation_chart(curve_key: str, future_curve: tuple,
                             current_risk: float, critical: float) -> bytes:
    """Gelecek risk eğrisini PNG (bytes) olarak çiz. Agg backend + BytesIO →
    dosya yolu YOK (Unicode path gotcha'sı oluşmaz)."""
    import matplotlib
    matplotlib.use("Agg")  # Streamlit thread — zorunlu
    import matplotlib.pyplot as plt
    from io import BytesIO

    BEDAS_BLUE = "#003a8c"
    months = list(range(len(future_curve) + 1))          # 0 = şimdi
    values = [current_risk] + list(future_curve)

    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=110)
    ax.plot(months, values, color=BEDAS_BLUE, linewidth=2.2, zorder=3)
    ax.axhline(critical, color="#ff3b30", linestyle="--", linewidth=1.4,
               zorder=2, label=f"Kritik esik ({critical:.0f})")

    # Eşiği ilk geçtiği nokta — vurgu.
    cross = next((i for i, v in enumerate(values) if v >= critical), None)
    if cross is not None:
        ax.scatter([months[cross]], [values[cross]], color="#ff3b30", s=60,
                   zorder=4, edgecolors="white", linewidths=1.2)
        ax.annotate(f"~{cross} ay", (months[cross], values[cross]),
                    textcoords="offset points", xytext=(6, -14),
                    color="#ff3b30", fontsize=9, fontweight="bold")

    ax.scatter([0], [current_risk], color=BEDAS_BLUE, s=45, zorder=4,
               edgecolors="white", linewidths=1.2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(future_curve))
    ax.set_xlabel("Gelecek Aylar", fontsize=9)
    ax.set_ylabel("Risk Skoru (0-100)", fontsize=9)
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.6)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def render_predictive_panel(fault: pd.Series, theme: dict) -> None:
    """Kestirimci bakım paneli (Aşama 7.4-UI).

    Model teşhisi panelinin ALTINA genişletilebilir bölüm: tahmini kritik süre
    rozeti + gelecek risk eğrisi grafiği + zorunlu SENTETİK veri uyarısı.
    Graceful: model yok / veri eksik / exception → sessiz mesaj, çökme YOK.
    """
    BEDAS_BLUE = "#003a8c"

    with st.expander("📈 Kestirimci Bakım (Bozulma Tahmini)", expanded=False):
        risk_score = fault.get("risk_score")
        if risk_score is None or pd.isna(risk_score):
            st.markdown(_collapse_html(f"""
            <div style="color:{theme['text_secondary']}; font-size:0.9rem; padding:0.25rem 0;">
                Tahmin için yeterli veri yok (risk skoru mevcut değil).
            </div>
            """), unsafe_allow_html=True)
            return

        ts = fault.get("timestamp")
        start_iso = None
        if ts is not None and pd.notna(ts):
            try:
                start_iso = pd.to_datetime(ts).date().isoformat()
            except Exception:
                start_iso = None

        forecast = _cached_degradation_forecast(
            fault.get("fault_id_short", ""), float(risk_score),
            fault.get("peak_temp"), fault.get("delta_temp"),
            fault.get("detection_confidence"), start_iso,
        )

        if forecast is None:
            st.markdown(_collapse_html(f"""
            <div style="color:{theme['text_secondary']}; font-size:0.9rem; padding:0.25rem 0;">
                Tahmin modeli mevcut değil veya tahmin üretilemedi.
                (models/trained/predictive/lstm_degradation.pt bulunamadı.)
            </div>
            """), unsafe_allow_html=True)
            return

        mtc = forecast["months_to_critical"]
        est_date = forecast.get("estimated_critical_date")
        confidence = forecast.get("confidence", 0.0)

        # Üst rozet — tahmini kritik süre + renk kodu.
        if mtc == 0:
            badge_color = "#ff3b30"
            badge_main = "Ekipman ZATEN kritik risk bandında"
            badge_sub = ""
        elif mtc is None:
            badge_color = "#009a3a"
            badge_main = f"Ufuk içinde (~{forecast['horizon_months']} ay) kritik beklenmiyor"
            badge_sub = ""
        else:
            badge_color = ("#ff3b30" if mtc < 6 else "#ff9500" if mtc <= 18 else "#009a3a")
            badge_main = f"Tahmini kritik süre: ~{mtc} ay"
            if est_date:
                try:
                    d = pd.to_datetime(est_date)
                    badge_sub = f"Tahmini kritik tarih: {d.strftime('%m/%Y')}"
                except Exception:
                    badge_sub = ""
            else:
                badge_sub = ""

        sub_html = (f'<div style="font-size:0.85rem; color:{theme["text_secondary"]}; '
                    f'margin-top:4px;">{badge_sub}</div>') if badge_sub else ""

        st.markdown(_collapse_html(f"""
        <div style="font-family:-apple-system,'Segoe UI',sans-serif;">
            <div style="background:{badge_color}1A; border:1px solid {badge_color};
                        border-left:5px solid {badge_color}; border-radius:8px;
                        padding:12px 16px; margin-bottom:12px;">
                <div style="font-size:1.25rem; font-weight:700; color:{badge_color};">
                    {badge_main}
                </div>
                {sub_html}
                <div style="font-size:0.8rem; color:{theme['text_secondary']}; margin-top:6px;">
                    Mevcut risk: {forecast['current_risk']:.0f}/100 &nbsp;•&nbsp;
                    Model güveni: %{confidence * 100:.0f}
                </div>
            </div>
        </div>
        """), unsafe_allow_html=True)

        # Gelecek risk eğrisi grafiği.
        curve = forecast.get("future_risk_curve") or []
        if curve:
            try:
                png = _build_degradation_chart(
                    fault.get("fault_id_short", ""), tuple(curve),
                    float(forecast["current_risk"]), 75.0,
                )
                st.image(png, use_container_width=True)
            except Exception:  # noqa: BLE001
                st.markdown(_collapse_html(f"""
                <div style="color:{theme['text_secondary']}; font-size:0.85rem;">
                    Grafik oluşturulamadı.
                </div>
                """), unsafe_allow_html=True)

        # SENTETİK veri uyarısı — zorunlu (akademik dürüstlük).
        disclaimer = forecast.get("disclaimer") or (
            "Bu tahmin, saha boylamsal verisi olmadığından fiziksel bozulma "
            "modelinden eğitilmiş bir model ile üretilmiştir.")
        st.markdown(_collapse_html(f"""
        <div style="margin-top:10px; background:#fff8e1; border-left:4px solid #ff9500;
                    padding:10px 14px; border-radius:6px;">
            <strong style="color:#b26a00;">⚠️ Sentetik Veri Uyarısı:</strong>
            <div style="font-size:0.84rem; color:{theme['text_primary']}; margin-top:4px;">
                Bu tahmin, saha boylamsal verisi olmadığından fiziksel bozulma
                modelinden eğitilmiş bir model ile üretilmiştir (mimari doğrulaması;
                gerçek saha verisiyle doğrulama gelecek çalışmadır).
            </div>
        </div>
        """), unsafe_allow_html=True)


def render_detail_panel(fault: pd.Series) -> None:
    """Seçili arıza için tam detay paneli (görüntüler + metadata + aksiyonlar)."""
    theme = get_current_theme()
    fault_dir = OUTPUTS_DIR / f"fault_{fault['fault_id_short']}"

    st.markdown(f"### 🔍 Arıza Detayı: `{fault['fault_id_short']}`")

    # Risk badge
    risk_color_map = {
        "critical": theme["danger"],
        "high": theme["orange"],
        "medium": theme["warning"],
        "low": theme["success"],
    }
    risk_color = risk_color_map.get(fault['risk_level'], theme['text_secondary'])

    st.markdown(f"""
    <div style="background:{risk_color}; color:white; padding:12px 20px;
                border-radius:8px; margin-bottom:1rem; font-weight:600;">
        {fault['risk_level_tr'].upper()} RİSK — Skor: {fault['risk_score']:.1f}/100
    </div>
    """, unsafe_allow_html=True)

    # Görüntüler — 3 sütun
    st.markdown("#### 📷 Görüntüler")
    st.caption("💡 Görüntüye tıklayarak büyütebilir, alttaki butonla indirebilirsiniz.")

    img_col1, img_col2, img_col3 = st.columns(3)

    rgb_original = fault_dir / "01_rgb_original.jpg"
    rgb_annotated = fault_dir / "02_rgb_annotated.jpg"
    thermal_annotated = fault_dir / "04_thermal_annotated.png"

    fault_id_short = fault['fault_id_short']

    def render_image_with_download(col, image_path: Path, label: str, dl_name: str) -> None:
        """Tek görüntü kolonu — görüntü + indirme butonu."""
        with col:
            st.markdown(f"**{label}**")
            if image_path.exists():
                st.image(str(image_path), width='stretch')
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                # Kompakt indir butonu — Apple-style download arrow, secondary
                st.download_button(
                    label="⬇  İndir",
                    data=img_bytes,
                    file_name=dl_name,
                    mime="image/jpeg" if image_path.suffix == ".jpg" else "image/png",
                    key=f"dl_{label}_{image_path.stem}_{fault['fault_id'][:8]}",
                    type="secondary",  # NOT primary — natural küçük boyut
                )
            else:
                st.info("📷 Görüntü mevcut değil")

    render_image_with_download(img_col1, rgb_original, "Orijinal RGB",
                               f"{fault_id_short}_rgb_original.jpg")
    render_image_with_download(img_col2, rgb_annotated, "Tespit (BBox)",
                               f"{fault_id_short}_rgb_annotated.jpg")
    render_image_with_download(img_col3, thermal_annotated, "Termal (Hotspot)",
                               f"{fault_id_short}_thermal_annotated.png")

    st.markdown("---")

    # ==== TEMEL BİLGİLER + KONUM & TERMAL ====
    # İki kartı TEK HTML block'ta CSS Grid ile render et → eşit yükseklik garanti
    conf_html = ""
    if pd.notna(fault['detection_confidence']):
        conf_html = (f'<p style="margin:6px 0;"><strong>Model Güveni:</strong> '
                     f"%{fault['detection_confidence'] * 100:.1f}</p>")

    if pd.notna(fault['latitude']):
        konum_html = (f'<p style="margin:6px 0;"><strong>Konum:</strong> '
                      f"{fault['latitude']:.6f}°N, {fault['longitude']:.6f}°E</p>")
    else:
        konum_html = '<p style="margin:6px 0;"><strong>Konum:</strong> —</p>'

    thermal_html = ""
    if pd.notna(fault['peak_temp']):
        peak = float(fault['peak_temp'])
        delta_val = float(fault['delta_temp']) if pd.notna(fault['delta_temp']) else 0.0
        assessment = assess_thermal(peak, delta_val)
        thermal_html = f"""
        <p style="margin:6px 0;"><strong>Peak Sıcaklık:</strong> {peak:.1f}°C
            <span style="background:{assessment.color}; color:white; padding:2px 10px;
                         border-radius:6px; font-size:0.78rem; font-weight:600; margin-left:8px;">
                {assessment.emoji} {assessment.label_tr.upper()}
            </span>
        </p>
        <p style="margin:6px 0;"><strong>Δ Fark:</strong> {delta_val:.1f}°C</p>
        <div style="background:{assessment.color}26; border-left:3px solid {assessment.color};
                    padding:8px 12px; border-radius:6px; margin-top:10px;">
            <strong style="color:{assessment.color};">Termal Değerlendirme: {assessment.label_tr}</strong>
            <p style="margin:4px 0 0 0; color:{theme['text_primary']}; font-size:0.85rem;">
                {get_thermal_recommendation(assessment.final_status)}
            </p>
        </div>
        """
    else:
        thermal_html = '<p style="margin:6px 0;"><strong>Termal:</strong> —</p>'

    notes_html = ""
    if fault['notes']:
        notes_html = f"""
        <div style="margin-top:14px; padding-top:14px;
                    border-top:1px solid {theme['border_light']};">
            <strong style="color:{theme['text_primary']};">Notlar:</strong><br>
            <span style="color:{theme['text_secondary']};">{fault['notes']}</span>
        </div>
        """

    timestamp_str = pd.to_datetime(fault['timestamp']).strftime('%d.%m.%Y %H:%M')

    detail_grid_html = f"""
    <div style="display:grid; grid-template-columns:1fr 1fr;
                gap:1rem; margin:1rem 0;">

        <div style="background:{theme['bg_tertiary']}; padding:1rem 1.25rem;
                    border-radius:12px;
                    border:1px solid {theme['border_light']};
                    border-left:4px solid {theme['accent_primary']};
                    min-height:280px;
                    display:flex;
                    flex-direction:column;">
            <h4 style="margin:0 0 12px 0; color:{theme['text_primary']};
                       font-size:1rem; font-weight:600;">
                📋 Temel Bilgiler
            </h4>
            <div style="flex:1; color:{theme['text_primary']};">
                <p style="margin:6px 0;"><strong>Tespit Tarihi:</strong> {timestamp_str}</p>
                <p style="margin:6px 0;"><strong>Uçuş:</strong> {fault['flight_label']}</p>
                <p style="margin:6px 0;"><strong>Arıza Tipi:</strong> {fault['fault_type_label']}</p>
                <p style="margin:6px 0;"><strong>Ekipman:</strong> {fault['equipment_label']}</p>
                <p style="margin:6px 0;"><strong>Durum:</strong> {fault['status_tr']}</p>
                {conf_html}
            </div>
        </div>

        <div style="background:{theme['bg_tertiary']}; padding:1rem 1.25rem;
                    border-radius:12px;
                    border:1px solid {theme['border_light']};
                    border-left:4px solid {theme['orange']};
                    min-height:280px;
                    display:flex;
                    flex-direction:column;">
            <h4 style="margin:0 0 12px 0; color:{theme['text_primary']};
                       font-size:1rem; font-weight:600;">
                📍 Konum & Termal
            </h4>
            <div style="flex:1; color:{theme['text_primary']};">
                {konum_html}
                {thermal_html}
                {notes_html}
            </div>
        </div>

    </div>
    """
    st.markdown(_collapse_html(detail_grid_html), unsafe_allow_html=True)

    # Risk skoru şeffaflık paneli (Stage 7.1A) — AI Teşhisi'nden ÖNCE
    render_risk_explanation_panel(fault, theme)

    # AI Teşhisi (Stage 2 defect classifier + Grad-CAM)
    render_ai_diagnosis_panel(fault, theme)

    # Model Karar Analizi — XAI yorum katmanı (AI Teşhisi'nin ALTINA, Aşama 7.1B)
    render_model_explainability_panel(fault, theme)

    # Kestirimci Bakım — bozulma tahmini (Model Teşhisi'nin ALTINA, Aşama 7.4-UI)
    render_predictive_panel(fault, theme)

    # ==== RAPOR İNDİR (PDF Saha Raporu) ====
    st.markdown("---")
    st.markdown(f'<h3 style="color:{theme["text_primary"]};">📄 Rapor İndir</h3>',
                unsafe_allow_html=True)
    col_pdf, col_empty = st.columns([1, 3])
    with col_pdf:
        pdf_bytes = _generate_field_report_pdf(fault['fault_id_short'])
        if pdf_bytes:
            st.download_button(
                label="📄 Saha Raporu (PDF)",
                data=pdf_bytes,
                file_name=f"HHATS_Saha_Raporu_{fault['fault_id_short']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"pdf_dl_{fault['fault_id_short']}",
            )
        else:
            st.button("📄 PDF (oluşturulamadı)", disabled=True,
                      use_container_width=True, key=f"pdf_dis_{fault['fault_id_short']}")

    st.markdown("---")

    st.markdown("### ⚡ Aksiyon Merkezi")

    # === DURUM YÖNETİMİ ===
    with st.container():
        st.markdown("#### 🎯 Durum Yönetimi")
        cur_status = fault['status_tr']
        status_color = {
            'Yeni': theme['accent_primary'],
            'İncelendi': theme['warning'],
            'Çözüldü': theme['success'],
            'Yanlış Alarm': theme['text_secondary'],
        }.get(cur_status, theme['text_secondary'])

        st.markdown(f"""
        <div style="background:{status_color}; color:white; padding:8px 16px;
                    border-radius:6px; display:inline-block; font-weight:600;
                    margin-bottom:0.8rem;">
            Mevcut Durum: {cur_status.upper()}
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            if st.button("✅ İncelendi olarak işaretle", key=f"st_rev_{fault['fault_id']}",
                         disabled=(fault['status'] == 'reviewed'), width='stretch'):
                if update_fault_status(fault['fault_id'], 'reviewed'):
                    st.success("Durum güncellendi: İncelendi"); st.rerun()
        with sc2:
            if st.button("🔧 Çözüldü olarak işaretle", key=f"st_res_{fault['fault_id']}",
                         disabled=(fault['status'] == 'resolved'), width='stretch'):
                if update_fault_status(fault['fault_id'], 'resolved'):
                    st.success("Durum güncellendi: Çözüldü"); st.rerun()
        with sc3:
            if st.button("❌ Yanlış Alarm", key=f"st_fp_{fault['fault_id']}",
                         disabled=(fault['status'] == 'false_positive'), width='stretch'):
                if update_fault_status(fault['fault_id'], 'false_positive'):
                    st.success("Durum güncellendi: Yanlış Alarm"); st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # === PERSONEL ATAMASI ===
    with st.container():
        st.markdown("#### 👤 Personel Ataması")

        personnel_df = load_personnel_list()
        assignment = load_fault_assignment(fault['fault_id'])
        current_pid = assignment.get('assigned_to_id')
        current_priority = assignment.get('priority', 'normal')

        # Mevcut atama bilgisi
        if current_pid:
            st.markdown(f"""
            <div style="background:{theme['bg_tertiary']}; border:1px solid {theme['border_light']};
                        border-left:4px solid {theme['accent_primary']}; padding:10px 14px;
                        border-radius:6px; margin-bottom:0.8rem;">
                <strong>Atanan:</strong> {assignment['assigned_to_name']}
                ({assignment['assigned_to_role']} • {assignment['assigned_to_team']})<br>
                <strong>Öncelik:</strong> {current_priority.upper()}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("👤 Bu arıza henüz kimseye atanmamış.")

        # Yeni atama formu
        ac1, ac2 = st.columns([2, 1])
        with ac1:
            personnel_options = [(None, "— Boş — (atamayı kaldır)")] + [
                (row['id'], f"{row['name']} • {row['role']} • {row['team']}")
                for _, row in personnel_df.iterrows()
            ]

            # Mevcut atamayı default seçili yap
            default_idx = 0
            if current_pid:
                for i, (pid, _) in enumerate(personnel_options):
                    if pid == current_pid:
                        default_idx = i
                        break

            new_assignment = st.selectbox(
                "Personel seç",
                options=[opt[0] for opt in personnel_options],
                format_func=lambda x: dict(personnel_options)[x],
                index=default_idx,
                key=f"assign_select_{fault['fault_id']}",
            )
        with ac2:
            new_priority = st.selectbox(
                "Öncelik",
                options=['low', 'normal', 'urgent'],
                format_func=lambda x: {'low': 'Düşük', 'normal': 'Normal', 'urgent': 'ACİL'}[x],
                index=['low', 'normal', 'urgent'].index(current_priority),
                key=f"prio_{fault['fault_id']}",
            )

        if st.button("💾 Atama Kaydet", key=f"save_assign_{fault['fault_id']}", width='stretch'):
            try:
                db = FaultDatabase(str(DB_PATH))
                if db.assign_fault(fault['fault_id'], new_assignment, new_priority):
                    st.cache_data.clear()
                    st.success("✓ Atama güncellendi.")
                    st.rerun()
                else:
                    st.error("Atama kaydedilemedi.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Hata: {e}")

        # Yeni personel ekleme - expander
        with st.expander("➕ Yeni Personel Ekle"):
            np_c1, np_c2 = st.columns(2)
            with np_c1:
                new_name = st.text_input("İsim", key=f"np_name_{fault['fault_id']}")
                new_role = st.text_input("Rol (örn: Saha Mühendisi)",
                                         key=f"np_role_{fault['fault_id']}")
            with np_c2:
                new_team = st.text_input("Ekip (örn: Kocaeli Ekibi)",
                                         key=f"np_team_{fault['fault_id']}")

            if st.button("➕ Personel Ekle", key=f"np_add_{fault['fault_id']}", width='stretch'):
                if not new_name.strip():
                    st.error("İsim boş olamaz.")
                else:
                    try:
                        db = FaultDatabase(str(DB_PATH))
                        pid = db.add_personnel(new_name.strip(), new_role.strip(), new_team.strip())
                        st.cache_data.clear()
                        st.success(f"✓ Personel eklendi (ID={pid}).")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Hata: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # === İNCELEME NOTLARI ===
    with st.container():
        st.markdown("#### 📝 İnceleme Notları")

        existing_notes = load_fault_notes(fault['fault_id'])

        if existing_notes:
            st.caption(f"{len(existing_notes)} not mevcut")
            for note in existing_notes:
                note_type_label = {
                    'technical': '🔧 Teknik',
                    'field': '🌍 Saha',
                    'management': '👔 Yönetim',
                }.get(note['type'], note['type'])

                note_color = {
                    'technical': theme['accent_primary'],
                    'field': theme['warning'],
                    'management': theme['success'],
                }.get(note['type'], theme['text_secondary'])

                created_str = (pd.to_datetime(note['created_at']).strftime('%d.%m.%Y %H:%M')
                               if note['created_at'] else '—')

                st.markdown(f"""
                <div style="background:{theme['bg_tertiary']}; border:1px solid {theme['border_light']};
                            border-left:4px solid {note_color}; padding:10px 14px;
                            border-radius:6px; margin-bottom:0.6rem;">
                    <div style="display:flex; justify-content:space-between; align-items:center;
                                margin-bottom:0.4rem;">
                        <strong style="color:{note_color};">{note_type_label}</strong>
                        <span style="color:{theme['text_secondary']}; font-size:0.85rem;">
                            {note['author']} • {created_str}
                        </span>
                    </div>
                    <div style="color:{theme['text_primary']};">{note['content']}</div>
                </div>
                """, unsafe_allow_html=True)

                # Sil butonu (her notun yanında)
                if st.button("🗑️ Notu sil", key=f"del_note_{note['id']}"):
                    try:
                        db = FaultDatabase(str(DB_PATH))
                        if db.delete_note(note['id']):
                            st.cache_data.clear()
                            st.success("Not silindi.")
                            st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Hata: {e}")
        else:
            st.caption("Henüz not eklenmemiş.")

        # Yeni not formu
        st.markdown("**Yeni Not Ekle:**")
        new_note_content = st.text_area(
            "İçerik",
            placeholder="Notu buraya yazın...",
            key=f"new_note_{fault['fault_id']}",
            height=100,
        )

        nc1, nc2, nc3 = st.columns([1, 1, 1])
        with nc1:
            new_note_type = st.selectbox(
                "Not Tipi",
                options=['technical', 'field', 'management'],
                format_func=lambda x: {'technical': '🔧 Teknik', 'field': '🌍 Saha',
                                       'management': '👔 Yönetim'}[x],
                key=f"new_note_type_{fault['fault_id']}",
            )
        with nc2:
            new_note_author = st.text_input(
                "Yazan",
                value="Ömer Kürşat Canbaz",
                key=f"new_note_author_{fault['fault_id']}",
            )
        with nc3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("📝 Not Ekle", key=f"add_note_{fault['fault_id']}", width='stretch'):
                if not new_note_content.strip():
                    st.error("Not içeriği boş olamaz.")
                else:
                    try:
                        db = FaultDatabase(str(DB_PATH))
                        note_id = db.add_note(
                            fault['fault_id'], new_note_type,
                            new_note_content.strip(), new_note_author.strip() or "Anonim",
                        )
                        st.cache_data.clear()
                        st.success(f"✓ Not eklendi (ID={note_id}).")
                        st.rerun()
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Hata: {e}")

    st.markdown("<br>", unsafe_allow_html=True)

    # === DİĞER İŞLEMLER ===
    with st.container():
        st.markdown("#### 🗂️ Diğer İşlemler")

        da_c1, da_c2 = st.columns(2)
        with da_c1:
            # Görüntü dosya yolu bilgisi (file:/// linki tarayıcı güvenliği nedeniyle kaldırıldı)
            st.caption(f"📁 Görüntü dosyaları: outputs/fault_files/fault_{fault_id_short}")

        with da_c2:
            # PASİFE ALMA (soft delete — veri korunur, listeden gizlenir)
            confirm_key = f"confirm_deactivate_{fault['fault_id']}"
            if confirm_key not in st.session_state:
                st.session_state[confirm_key] = False

            if not st.session_state[confirm_key]:
                if st.button("⏸️ Bu Arıza Kaydını Pasife Al",
                             key=f"deact_btn_{fault['fault_id']}",
                             width='stretch',
                             type="secondary"):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.warning("⚠️ Bu kayıt listeden gizlenecek. Veri silinmez, tekrar aktif edilebilir.")
                cd_c1, cd_c2 = st.columns(2)
                with cd_c1:
                    if st.button("✅ Evet, Pasife Al",
                                 key=f"yes_deact_{fault['fault_id']}",
                                 width='stretch', type="primary"):
                        try:
                            db = FaultDatabase(str(DB_PATH))
                            if db.deactivate_fault(fault['fault_id']):
                                st.cache_data.clear()
                                st.session_state[confirm_key] = False
                                st.success("Kayıt pasife alındı. Veri sistemde korunuyor.")
                                st.rerun()
                        except Exception as e:  # noqa: BLE001
                            st.error(f"Hata: {e}")
                with cd_c2:
                    if st.button("❌ Vazgeç",
                                 key=f"no_deact_{fault['fault_id']}",
                                 width='stretch'):
                        st.session_state[confirm_key] = False
                        st.rerun()


def main() -> None:
    theme = get_current_theme()

    # ==== SATIR TIKLAMA / KAPATMA YAKALAMA ====
    # Tablo satırındaki <a href="?fault_detail=ID"> → o arızanın detayını aç.
    fault_detail_id = st.query_params.get("fault_detail", None)
    if fault_detail_id:
        st.query_params.clear()  # URL'i temiz tut, tekrar tetiklenmesin
        st.session_state['faults_focus_id'] = fault_detail_id
        st.rerun()

    # "✕ Detayı Kapat" linki (?close_detail=1) → detay panelini kapat.
    if st.query_params.get("close_detail", None):
        st.query_params.clear()
        st.session_state.pop('faults_focus_id', None)
        st.rerun()

    st.markdown('<h1 class="main-title">🚨 Arıza Yönetimi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Detaylı arıza listesi, filtreleme, görüntü inceleme</p>',
                unsafe_allow_html=True)

    df = load_all_faults()
    if df.empty:
        st.warning("⚠️ Hiç arıza kaydı bulunamadı.")
        return

    # NOT: faults_focus_id artık KALICI bir "şu detayı göster" bayrağıdır
    # (Map marker / Dashboard tablo / tablo satırı tıklaması hepsi bunu set eder).
    # Detay panel bölümünde okunur; "Kapat" ile temizlenir.

    THERMAL_OPTS = ['normal', 'warning', 'critical']  # termal filtre seçenekleri

    # ==== FİLTRELER ====
    with st.expander("🔍 Filtreler ve Arama", expanded=True):
        # Satır 1
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_text = st.text_input(
                "🔎 Arama",
                placeholder="Fault ID, ekipman, arıza tipi...",
                key="faults_search",
            )
        with f2:
            # Dashboard'dan yönlendirme varsa default filtreyi uygula (tek seferlik)
            default_risk = st.session_state.get(
                'faults_default_risk', ['critical', 'high', 'medium', 'low'])
            if 'faults_default_risk' in st.session_state:
                del st.session_state['faults_default_risk']
            risk_filter = st.multiselect(
                "Risk Seviyesi",
                options=['critical', 'high', 'medium', 'low'],
                default=default_risk,
                format_func=lambda x: RISK_TR.get(x, x),
            )
        with f3:
            # Dashboard'dan durum yönlendirmesi (tek seferlik)
            default_status = st.session_state.get(
                'faults_default_status', ['new', 'reviewed', 'resolved', 'false_positive'])
            if 'faults_default_status' in st.session_state:
                del st.session_state['faults_default_status']
            status_filter = st.multiselect(
                "Durum",
                options=['new', 'reviewed', 'resolved', 'false_positive'],
                default=default_status,
                format_func=lambda x: STATUS_TR.get(x, x),
            )

        # Satır 2
        f4, f5, f6, f7 = st.columns(4)
        with f4:
            all_flights = sorted(df['flight_id'].unique().tolist())
            # Dashboard grafiğinden uçuş yönlendirmesi (tek seferlik)
            default_flight = st.session_state.pop('faults_default_flight', 'Hepsi')
            flight_filter = st.selectbox(
                "Uçuş",
                options=['Hepsi'] + all_flights,
                index=0 if default_flight == 'Hepsi' else
                      (all_flights.index(default_flight) + 1
                       if default_flight in all_flights else 0),
                format_func=lambda x: "Hepsi" if x == 'Hepsi' else FLIGHT_LABELS.get(x, x),
            )
        with f5:
            all_types = sorted(df['fault_type'].unique().tolist())
            # Dashboard'dan arıza tipi yönlendirmesi (tek seferlik)
            default_type = st.session_state.get('faults_default_type', all_types)
            if 'faults_default_type' in st.session_state:
                del st.session_state['faults_default_type']
            # Geçersiz tipleri filtrele (DB'de olmayan tip session'da kalmasın)
            default_type = [t for t in default_type if t in all_types]
            type_filter = st.multiselect(
                "Arıza Tipi",
                options=all_types,
                default=default_type,
                format_func=lambda x: FAULT_TYPE_LABELS.get(x, x),
            )
        with f6:
            sort_option = st.selectbox(
                "Sıralama",
                options=['Risk Skoru ↓', 'Risk Skoru ↑', 'Tarih (Yeni)', 'Tarih (Eski)'],
            )
        with f7:
            # Dashboard termal KPI'sinden yönlendirme (tek seferlik)
            default_thermal = st.session_state.get('faults_default_thermal', THERMAL_OPTS)
            if 'faults_default_thermal' in st.session_state:
                del st.session_state['faults_default_thermal']
            default_thermal = [t for t in default_thermal if t in THERMAL_OPTS]
            thermal_filter = st.multiselect(
                "Termal Durum",
                options=THERMAL_OPTS,
                default=default_thermal,
                format_func=lambda x: f"{STATUS_EMOJI[ThermalStatus(x)]} {STATUS_LABELS_TR[ThermalStatus(x)]}",
            )

    # ==== FİLTRELERİ UYGULA ====
    filtered = df.copy()
    if search_text:
        # Türkçe-duyarlı: hem aranan metni hem önceden hesaplanmış search_blob'u
        # ASCII'ye indirgeyerek karşılaştır (regex=False → özel karakterler literal).
        query = _tr_normalize(search_text)
        filtered = filtered[filtered['search_blob'].str.contains(query, regex=False, na=False)]

    filtered = filtered[filtered['risk_level'].isin(risk_filter)]
    filtered = filtered[filtered['status'].isin(status_filter)]
    filtered = filtered[filtered['fault_type'].isin(type_filter)]
    if flight_filter != 'Hepsi':
        filtered = filtered[filtered['flight_id'] == flight_filter]
    # Termal filtre: yalnızca daraltıldığında uygula (tam set seçiliyken termal verisi
    # olmayan kayıtlar da görünsün diye filtreleme yapma)
    if set(thermal_filter) != set(THERMAL_OPTS):
        filtered = filtered[filtered['thermal_status'].isin(thermal_filter)]

    # Sıralama
    if sort_option == 'Risk Skoru ↓':
        filtered = filtered.sort_values('risk_score', ascending=False)
    elif sort_option == 'Risk Skoru ↑':
        filtered = filtered.sort_values('risk_score', ascending=True)
    elif sort_option == 'Tarih (Yeni)':
        filtered = filtered.sort_values('timestamp', ascending=False)
    else:
        filtered = filtered.sort_values('timestamp', ascending=True)

    filtered = filtered.reset_index(drop=True)  # iloc[selected_idx] satır seçimi için

    # ==== ÖZET METRİKLER ====
    st.markdown("<br>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        render_metric_card("Görüntülenen", f"{len(filtered)}", theme['accent_primary'])
    with mc2:
        render_metric_card("Kritik", f"{(filtered['risk_level'] == 'critical').sum()}", theme['danger'])
    with mc3:
        unresolved = filtered[~filtered['status'].isin(['resolved', 'false_positive'])]
        render_metric_card("Çözülmemiş", f"{len(unresolved)}", theme['orange'])
    with mc4:
        now = datetime.now()
        ts = pd.to_datetime(filtered['timestamp'])
        this_month = filtered[(ts.dt.month == now.month) & (ts.dt.year == now.year)]
        render_metric_card("Bu Ay", f"{len(this_month)}", theme['success'])

    st.markdown("<br>", unsafe_allow_html=True)

    # ==== EXCEL EXPORT (filtrelenmiş veri) ====
    if not filtered.empty:
        export_col1, export_col2, export_col3 = st.columns([2, 1, 2])
        with export_col2:
            excel_bytes = export_faults_to_excel(filtered)
            st.download_button(
                label="📥 Excel İndir",
                data=excel_bytes,
                file_name=f"hhats_arizalar_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_export_faults",
                width='stretch',
            )

    # ==== LİSTE TABLOSU ====
    st.markdown("### 📋 Arıza Listesi")

    if filtered.empty:
        st.warning("Filtrelere uyan kayıt bulunmuyor.")
        return

    # Pagination — >50 kayıt varsa sayfala
    PAGE_SIZE = 50
    total = len(filtered)
    page_df = filtered
    page = 0
    if total > PAGE_SIZE:
        n_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
        pcol1, pcol2 = st.columns([1, 3])
        with pcol1:
            page = st.number_input(
                f"Sayfa (1-{n_pages})", min_value=1, max_value=n_pages, value=1,
                key="faults_page",
            ) - 1
        start = page * PAGE_SIZE
        end = start + PAGE_SIZE
        page_df = filtered.iloc[start:end]
        with pcol2:
            st.caption(f"Toplam {total} kayıt · {start + 1}-{min(end, total)} arası gösteriliyor")

    # Tabloyu hazırla — pure-HTML tablo (st.dataframe DEĞİL → tema GARANTİ uygulanır)
    disp = page_df[['fault_id_short', 'timestamp', 'risk_level_tr', 'risk_score',
                    'fault_type_label', 'equipment_label', 'flight_label', 'status_tr']].copy()
    disp['timestamp'] = pd.to_datetime(disp['timestamp']).dt.strftime('%d.%m.%Y %H:%M')
    disp.columns = ['ID', 'Tarih', 'Risk', 'Skor', 'Tip', 'Ekipman', 'Uçuş', 'Durum']
    disp['_fault_id_full'] = page_df['fault_id'].values  # gizli: satır linki için tam id

    # Termal durum etiketi ("🟡 Dikkat" / "—") — load_all_faults'tan gelen thermal_status'tan
    _THERMAL_TR = {
        'normal': f"{STATUS_EMOJI[ThermalStatus.NORMAL]} {STATUS_LABELS_TR[ThermalStatus.NORMAL]}",
        'warning': f"{STATUS_EMOJI[ThermalStatus.WARNING]} {STATUS_LABELS_TR[ThermalStatus.WARNING]}",
        'critical': f"{STATUS_EMOJI[ThermalStatus.CRITICAL]} {STATUS_LABELS_TR[ThermalStatus.CRITICAL]}",
    }
    disp['Termal'] = [
        _THERMAL_TR.get(s, "—") for s in page_df['thermal_status'].values
    ]

    st.caption("💡 Detayını görmek için tablodaki bir satıra tıklayın.")
    st.markdown(render_faults_table_html(disp, theme), unsafe_allow_html=True)

    # ==== DETAY PANELİ (satır tıklanınca faults_focus_id set olur) ====
    focus_id = st.session_state.get('faults_focus_id', None)
    if focus_id:
        matches = df[df['fault_id'] == focus_id]
        if not matches.empty:
            # "✕ Detayı Kapat" çubuğu (link → ?close_detail=1)
            st.markdown(_collapse_html(f"""
            <div style="display:flex; justify-content:flex-end; margin:1.5rem 0 0 0;">
                <a href="?close_detail=1" target="_self"
                   style="color:{theme['text_secondary']}; text-decoration:none;
                          font-size:0.9rem; padding:6px 14px;
                          border:1px solid {theme['border_light']};
                          border-radius:8px; background:{theme['bg_tertiary']};">
                    ✕ Detayı Kapat
                </a>
            </div>
            """), unsafe_allow_html=True)
            st.markdown("---")
            render_detail_panel(matches.iloc[0])
        else:
            # Geçersiz/silinmiş fault_id → state'i temizle
            st.session_state.pop('faults_focus_id', None)
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("💡 Detayları görmek için tablodaki bir satıra tıklayın.")
    else:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 Detayları görmek için tablodaki bir satıra tıklayın.")


main()
