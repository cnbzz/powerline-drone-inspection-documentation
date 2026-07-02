"""Rapor Merkezi — tüm arızaların PDF saha raporlarını listeleme, filtreleme,
tekli + toplu (ZIP) indirme.

Mevcut PDF üreticisini (``src/dashboard/utils/pdf/field_report.py``
``generate_field_report``) KULLANIR — yeniden yazmaz. Veri kaynağı
``outputs/db/faults.db``. Diğer sayfaların mantığına dokunmaz.

Not: veri yükleyici, 3_Faults.py'deki ``load_all_faults`` ile aynı kolonları
üretir (o modülü import etmek sayfayı çalıştırırdı) — kodtabanındaki
sayfa-başına-loader desenine uygun (bkz. 1_Dashboard, 2_Map).
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Rapor Merkezi - HHATS", page_icon="📄", layout="wide")

from src.dashboard.app import DB_PATH, get_current_theme, load_css, render_sidebar  # noqa: E402

load_css()

from src.auth.session import require_login  # noqa: E402

require_login()  # giriş kapısı — giriş yoksa login formu + st.stop()
render_sidebar()

from src.fault.database import FaultDatabase  # noqa: E402
from src.thermal.thresholds import assess_thermal  # noqa: E402

# BEDAŞ kurumsal + risk renkleri
BEDAS_BLUE = "#003a8c"
BEDAS_GREEN = "#009a3a"

# --------------------------------------------------------------------------- #
# Türkçe etiket sözlükleri (3_Faults ile birebir)                              #
# --------------------------------------------------------------------------- #
RISK_TR = {"critical": "Kritik", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}
RISK_COLORS = {"critical": "#ff3b30", "high": "#ff9500", "medium": "#ffcc00",
               "low": "#34c759"}
STATUS_TR = {"new": "Yeni", "reviewed": "İncelendi", "resolved": "Çözüldü",
             "false_positive": "Yanlış Alarm"}
FAULT_TYPE_LABELS = {
    "thermal_hotspot": "Termal Anomali", "visual_defect": "Görsel Hasar",
    "equipment_detected": "Normal Ekipman", "combined": "Çoklu Arıza",
    "vegetation_encroachment": "Bitki Örtüsü", "unknown": "Belirsiz",
}
CLASS_NAMES_TR = {
    "insulator_polymer": "Polimer İzolatör", "insulator_glass": "Cam İzolatör",
    "insulator_porcelain": "Porselen İzolatör", "insulator_generic": "İzolatör",
    "tower": "Direk", "spacer": "Spacer", "damper": "Damper",
    "tower_id_plate": "Direk Plakası",
}
FLIGHT_LABELS = {
    "flight_2026_05_01_kocaeli": "Kocaeli-Gebze (01.05.2026)",
    "flight_2026_05_08_sakarya": "Sakarya-Adapazarı (08.05.2026)",
    "flight_2026_05_15_bursa": "Bursa-İnegöl (15.05.2026)",
    "flight_2026_05_22_istanbul": "İstanbul-Çatalca (22.05.2026)",
}
THERMAL_TR = {"critical": "Kritik", "warning": "Dikkat", "normal": "Normal"}
THERMAL_COLORS = {"critical": "#ff3b30", "warning": "#ff9500", "normal": "#34c759"}

ZIP_LIMIT = 50   # toplu üretim üst sınırı (performans)
PAGE_SIZE = 10   # tekli indirme sayfa başına satır

# Türkçe-duyarlı arama (dotted-İ gotcha): tüm TR karakterleri ASCII'ye indir.
_TR_MAP = str.maketrans({
    "İ": "i", "I": "i", "ı": "i", "Ş": "s", "ş": "s", "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u", "Ö": "o", "ö": "o", "Ç": "c", "ç": "c",
})


def _tr_normalize(text: str) -> str:
    if not text:
        return ""
    return text.translate(_TR_MAP).lower()


# --------------------------------------------------------------------------- #
# Veri yükleme (3_Faults.load_all_faults ile aynı kolonlar)                     #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30)
def load_all_faults() -> pd.DataFrame:
    """Tüm arıza kayıtlarını generate_field_report'un beklediği kolonlarla döndür."""
    db = FaultDatabase(str(DB_PATH))
    rows = []
    for r in db.list_all(limit=10000):
        equipment_label = CLASS_NAMES_TR.get(r.equipment_class, r.equipment_class or "—")
        fault_type_label = FAULT_TYPE_LABELS.get(r.fault_type.value, r.fault_type.value)
        thermal_status = (
            assess_thermal(r.thermal.peak_temp_c,
                           r.thermal.temperature_delta_c).final_status.value
            if r.thermal else None
        )
        rows.append({
            "fault_id": r.fault_id,
            "fault_id_short": r.fault_id[:8],
            "search_blob": _tr_normalize(
                f"{r.fault_id[:8]} {equipment_label} {fault_type_label} {r.notes or ''}"
            ),
            "timestamp": r.timestamp,
            "flight_label": FLIGHT_LABELS.get(r.inspection_flight_id,
                                              r.inspection_flight_id or "—"),
            "fault_type_label": fault_type_label,
            "equipment_class": r.equipment_class,
            "equipment_label": equipment_label,
            "risk_level": r.risk_level.value,
            "risk_level_tr": RISK_TR.get(r.risk_level.value, r.risk_level.value),
            "risk_score": r.risk_score,
            "status_tr": STATUS_TR.get(r.status.value, r.status.value),
            "detection_confidence": r.detection_confidence,
            "latitude": r.gps.latitude if r.gps else None,
            "longitude": r.gps.longitude if r.gps else None,
            "peak_temp": r.thermal.peak_temp_c if r.thermal else None,
            "delta_temp": r.thermal.temperature_delta_c if r.thermal else None,
            "thermal_status": thermal_status,
        })
    return pd.DataFrame(rows)


@st.cache_data(ttl=300, show_spinner=False)
def _report_pdf(fault_id_short: str):
    """Bir arıza için PDF saha raporu (cache'li). Üretilemezse None."""
    try:
        from src.dashboard.utils.pdf.field_report import generate_field_report
    except Exception:  # noqa: BLE001
        return None
    df = load_all_faults()
    matches = df[df["fault_id_short"] == fault_id_short]
    if matches.empty:
        return None
    try:
        return generate_field_report(matches.iloc[0])
    except Exception:  # noqa: BLE001
        return None


def _risk_badge(level: str, label: str) -> str:
    color = RISK_COLORS.get(level, "#8e8e93")
    return (f"<span style='background:{color}; color:white; padding:2px 10px; "
            f"border-radius:6px; font-size:0.8rem; font-weight:600;'>{label}</span>")


def _thermal_cell(status) -> str:
    if not status or (isinstance(status, float) and pd.isna(status)):
        return "<span style='color:#8e8e93;'>—</span>"
    color = THERMAL_COLORS.get(status, "#8e8e93")
    return (f"<span style='color:{color}; font-weight:600;'>"
            f"{THERMAL_TR.get(status, status)}</span>")


# --------------------------------------------------------------------------- #
# Sayfa                                                                        #
# --------------------------------------------------------------------------- #
theme = get_current_theme()

st.markdown('<h1 class="main-title">📄 Rapor Merkezi</h1>', unsafe_allow_html=True)
st.markdown(
    f"<p class='subtitle' style='color:{theme['text_secondary']};'>"
    f"Tüm arızaların PDF saha raporlarını buradan filtreleyip tek tek veya toplu "
    f"(ZIP) indirebilirsiniz.</p>",
    unsafe_allow_html=True,
)

df = load_all_faults()

# Graceful: hiç arıza yok
if df.empty:
    st.info("📭 Rapor üretilecek arıza bulunamadı. Veritabanında kayıt yok.")
    st.stop()

# --- ÜST ÖZET KARTLARI (tek blok CSS-Grid) ---
total = len(df)
by_risk = df["risk_level"].value_counts().to_dict()
n_crit = int(by_risk.get("critical", 0))
n_high = int(by_risk.get("high", 0))
n_med = int(by_risk.get("medium", 0))
n_low = int(by_risk.get("low", 0))
n_thermal_crit = int((df["thermal_status"] == "critical").sum())

cards = [
    ("Toplam Arıza", total, BEDAS_BLUE),
    ("Kritik", n_crit, RISK_COLORS["critical"]),
    ("Yüksek", n_high, RISK_COLORS["high"]),
    ("Orta", n_med, RISK_COLORS["medium"]),
    ("Düşük", n_low, RISK_COLORS["low"]),
    ("Termal Kritik", n_thermal_crit, "#ff3b30"),
]
card_html = "".join(
    f"""<div style="background:{theme['bg_tertiary']}; border-radius:14px;
        padding:14px 18px; border-left:5px solid {color};
        box-shadow:0 1px 2px rgba(0,0,0,0.04);">
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.04em;
                    color:{theme['text_secondary']};">{label}</div>
        <div style="font-size:1.9rem; font-weight:700; color:{color};">{value}</div>
    </div>"""
    for label, value, color in cards
)
st.markdown(
    f"""<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:12px; margin:0.5rem 0 1.5rem 0;">{card_html}</div>""",
    unsafe_allow_html=True,
)

# --- FİLTRE ÇUBUĞU ---
st.markdown(f"<h3 style='color:{BEDAS_BLUE} !important;'>🔎 Filtrele</h3>",
            unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([1, 1, 2])
with fc1:
    risk_opt = st.selectbox("Risk Seviyesi",
                            ["Tümü", "critical", "high", "medium", "low"],
                            format_func=lambda x: "Tümü" if x == "Tümü" else RISK_TR[x],
                            key="rep_risk")
with fc2:
    equips = ["Tümü"] + sorted(df["equipment_label"].dropna().unique().tolist())
    equip_opt = st.selectbox("Ekipman Türü", equips, key="rep_equip")
with fc3:
    search = st.text_input("Ara (arıza ID / ekipman)", key="rep_search",
                           placeholder="ör. cam izolatör, a695...")

filtered = df.copy()
if risk_opt != "Tümü":
    filtered = filtered[filtered["risk_level"] == risk_opt]
if equip_opt != "Tümü":
    filtered = filtered[filtered["equipment_label"] == equip_opt]
if search and search.strip():
    q = _tr_normalize(search.strip())
    filtered = filtered[filtered["search_blob"].str.contains(q, na=False)]

st.markdown(
    f"<p style='color:{theme['text_secondary']};'>"
    f"<b>{len(filtered)}</b> / {total} arıza gösteriliyor.</p>",
    unsafe_allow_html=True,
)

# Graceful: filtre sonucu boş
if filtered.empty:
    st.info("🔍 Seçilen filtrelere uyan arıza yok. Filtreleri değiştirin.")
    st.stop()

# --- TOPLU İNDİRME (ZIP) ---
st.markdown(f"<h3 style='color:{BEDAS_BLUE} !important;'>📦 Toplu İndirme</h3>",
            unsafe_allow_html=True)

zip_subset = filtered.head(ZIP_LIMIT)
if len(filtered) > ZIP_LIMIT:
    st.warning(f"⚠️ {len(filtered)} arıza filtrelendi; performans için ilk "
               f"{ZIP_LIMIT} rapor ZIP'e eklenecek. Daha fazla filtreleyin.")

filter_sig = f"{risk_opt}|{equip_opt}|{_tr_normalize(search or '')}|{len(zip_subset)}"

col_zip1, col_zip2 = st.columns([1, 2])
with col_zip1:
    make_zip = st.button(f"🗜️ {len(zip_subset)} Raporu ZIP Hazırla",
                         key="rep_make_zip", use_container_width=True)

if make_zip:
    with st.spinner(f"{len(zip_subset)} PDF raporu üretiliyor..."):
        buf = io.BytesIO()
        failed = 0
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for _, row in zip_subset.iterrows():
                pdf = _report_pdf(row["fault_id_short"])
                if pdf is None:          # bir arıza patlarsa atla, devam et
                    failed += 1
                    continue
                zf.writestr(f"HHATS_Saha_Raporu_{row['fault_id_short']}.pdf", pdf)
        st.session_state["rep_zip"] = {
            "data": buf.getvalue(), "failed": failed,
            "n": len(zip_subset), "sig": filter_sig,
        }

zipinfo = st.session_state.get("rep_zip")
if zipinfo and zipinfo["sig"] == filter_sig:
    ok_n = zipinfo["n"] - zipinfo["failed"]
    st.download_button(
        f"⬇️ ZIP İndir ({ok_n} rapor)", data=zipinfo["data"],
        file_name="HHATS_Saha_Raporlari.zip", mime="application/zip",
        key="rep_zip_dl", use_container_width=False,
    )
    if zipinfo["failed"]:
        st.warning(f"{zipinfo['failed']} rapor üretilemedi ve ZIP'e eklenmedi "
                   f"(diğerleri hazır).")

# --- RAPOR TABLOSU (HTML, st.dataframe DEĞİL) ---
st.markdown(f"<h3 style='color:{BEDAS_BLUE} !important;'>📋 Rapor Listesi</h3>",
            unsafe_allow_html=True)

head_style = (f"text-align:left; padding:9px 12px; background:{theme['bg_primary']}; "
              f"color:{theme['text_primary']}; font-weight:600; "
              f"border-bottom:2px solid {theme['border_medium']};")
rows_html = ""
for _, row in filtered.iterrows():
    ts = row["timestamp"]
    ts_str = pd.to_datetime(ts).strftime("%d.%m.%Y %H:%M") if pd.notna(ts) else "—"
    rows_html += (
        f"<tr>"
        f"<td style='padding:8px 12px; border-bottom:1px solid {theme['border_light']}; "
        f"font-family:monospace; color:{theme['text_primary']};'>{row['fault_id_short']}</td>"
        f"<td style='padding:8px 12px; border-bottom:1px solid {theme['border_light']}; "
        f"color:{theme['text_primary']};'>{row['equipment_label']}</td>"
        f"<td style='padding:8px 12px; border-bottom:1px solid {theme['border_light']};'>"
        f"{_risk_badge(row['risk_level'], row['risk_level_tr'])}</td>"
        f"<td style='padding:8px 12px; border-bottom:1px solid {theme['border_light']};'>"
        f"{_thermal_cell(row['thermal_status'])}</td>"
        f"<td style='padding:8px 12px; border-bottom:1px solid {theme['border_light']}; "
        f"color:{theme['text_secondary']};'>{ts_str}</td>"
        f"</tr>"
    )
st.markdown(
    f"""<table style="width:100%; border-collapse:collapse; background:{theme['bg_tertiary']};
        border-radius:12px; overflow:hidden; border:1px solid {theme['border_light']};
        font-size:0.9rem;">
        <thead><tr>
            <th style="{head_style}">Arıza ID</th>
            <th style="{head_style}">Ekipman</th>
            <th style="{head_style}">Risk</th>
            <th style="{head_style}">Termal</th>
            <th style="{head_style}">Tarih</th>
        </tr></thead><tbody>{rows_html}</tbody></table>""",
    unsafe_allow_html=True,
)

# --- TEKLİ İNDİRME (sayfalı — her satırda İndir butonu) ---
st.markdown(f"<h3 style='color:{BEDAS_BLUE} !important; margin-top:1.5rem;'>"
            f"⬇️ Tekli PDF İndirme</h3>", unsafe_allow_html=True)
st.caption(f"Performans için sayfa başına {PAGE_SIZE} rapor listelenir; "
           f"yalnızca görünen sayfanın PDF'leri üretilir.")

n_pages = (len(filtered) + PAGE_SIZE - 1) // PAGE_SIZE
page = 1
if n_pages > 1:
    page = st.number_input("Sayfa", min_value=1, max_value=n_pages, value=1, step=1,
                           key="rep_page")
start = (int(page) - 1) * PAGE_SIZE
page_rows = filtered.iloc[start:start + PAGE_SIZE]

for _, row in page_rows.iterrows():
    c_id, c_eq, c_risk, c_btn = st.columns([1.2, 1.6, 1, 1.2])
    fid = row["fault_id_short"]
    with c_id:
        st.markdown(f"<div style='font-family:monospace; padding-top:6px; "
                    f"color:{theme['text_primary']};'>{fid}</div>",
                    unsafe_allow_html=True)
    with c_eq:
        st.markdown(f"<div style='padding-top:6px; color:{theme['text_primary']};'>"
                    f"{row['equipment_label']}</div>", unsafe_allow_html=True)
    with c_risk:
        st.markdown(f"<div style='padding-top:4px;'>"
                    f"{_risk_badge(row['risk_level'], row['risk_level_tr'])}</div>",
                    unsafe_allow_html=True)
    with c_btn:
        pdf = _report_pdf(fid)
        if pdf:
            st.download_button("📄 PDF İndir", data=pdf,
                               file_name=f"HHATS_Saha_Raporu_{fid}.pdf",
                               mime="application/pdf", key=f"rep_dl_{fid}",
                               use_container_width=True)
        else:
            st.button("📄 Üretilemedi", disabled=True, key=f"rep_dis_{fid}",
                      use_container_width=True)
