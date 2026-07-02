"""Harita sayfası - Folium ile interaktif arıza haritası."""

from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import folium
import pandas as pd
import streamlit as st
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

st.set_page_config(page_title="Harita - HHATS", page_icon="🗺️", layout="wide")

from src.dashboard.app import DB_PATH, RISK_COLORS, get_current_theme, load_css, render_sidebar
from src.fault.database import FaultDatabase

load_css()

from src.auth.session import require_login  # noqa: E402

require_login()  # giriş kapısı — giriş yoksa login formu + st.stop()
render_sidebar()

# Folium marker renk haritası (folium-compatible)
FOLIUM_RISK_COLORS = {
    "low": "green",
    "medium": "orange",
    "high": "red",
    "critical": "darkred",
}

RISK_TR = {"critical": "Kritik", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}

FAULT_TYPE_LABELS = {
    "thermal_hotspot": "Termal Anomali",
    "visual_defect": "Görsel Hasar",
    "equipment_detected": "Normal Ekipman",
    "combined": "Çoklu Arıza",
    "vegetation_encroachment": "Bitki Örtüsü",
    "unknown": "Belirsiz",
}

FLIGHT_LABELS = {
    "flight_2026_05_01_kocaeli": "Kocaeli-Gebze Hattı (01.05.2026)",
    "flight_2026_05_08_sakarya": "Sakarya-Adapazarı Hattı (08.05.2026)",
    "flight_2026_05_15_bursa": "Bursa-İnegöl Hattı (15.05.2026)",
    "flight_2026_05_22_istanbul": "İstanbul-Çatalca Hattı (22.05.2026)",
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


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """DB'den fault records, GPS olanları filtrele."""
    db = FaultDatabase(str(DB_PATH))
    rows = []
    for r in db.list_all(limit=10000):
        if r.gps is None:  # GPS yoksa haritada gösterilemez
            continue
        rows.append({
            "fault_id": r.fault_id,
            "fault_id_short": r.fault_id[:8],
            "timestamp": r.timestamp,
            "flight_id": r.inspection_flight_id or "—",
            "flight_label": FLIGHT_LABELS.get(r.inspection_flight_id, r.inspection_flight_id or "—"),
            "fault_type": r.fault_type.value,
            "fault_type_label": FAULT_TYPE_LABELS.get(r.fault_type.value, r.fault_type.value),
            "equipment_class": r.equipment_class or "—",
            "equipment_label": CLASS_NAMES_TR.get(r.equipment_class, r.equipment_class or "—"),
            "risk_level": r.risk_level.value,
            "risk_level_tr": RISK_TR.get(r.risk_level.value, r.risk_level.value),
            "risk_score": r.risk_score,
            "status": r.status.value,
            "latitude": r.gps.latitude,
            "longitude": r.gps.longitude,
            "altitude_m": r.gps.altitude_m,
            "peak_temp": r.thermal.peak_temp_c if r.thermal else None,
        })
    return pd.DataFrame(rows)


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


def render_metric_card_clickable(label: str, value: str, color: str,
                                 filter_value: list = None, key: str = None) -> None:
    """Doğrudan tıklanabilir metrik kartı (buton YOK) — harita risk filtresini değiştirir.

    Tıklamada ``?map_filter=<vals>&map_key=<key>`` set edilir; main() okuyup
    ``map_risk_filter`` state'ini atayıp rerun yapar.
    """
    theme = get_current_theme()
    if filter_value is not None and key:
        safe_filter = ",".join(filter_value)
        st.markdown(f"""
        <a href="?map_filter={safe_filter}&map_key={key}"
           target="_self" class="kpi-card-link">
            <div class="kpi-card-clickable" style="border-left: 4px solid {color};
                        padding:0.8rem 1rem; text-align:center; border-radius:8px;">
                <div style="color:{theme['text_secondary']}; font-size:0.75rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.5px;">{label}</div>
                <div style="color:{color}; font-size:1.6rem; font-weight:700;
                            margin-top:0.2rem;">{value}</div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    else:
        render_metric_card(label, value, color)


def _collapse_html(html: str) -> str:
    """st.markdown render fix: baştaki girintiyi sil, boş satırları at.

    Streamlit st.markdown'ı Python-Markdown ile işler; 4+ boşlukla başlayan
    satır 'indented code block' sayılıp escape edilir (HTML çıplak metin görünür).
    Her satırı strip edip boş satırları atınca tek bitişik HTML bloğu kalır →
    HTML olarak render edilir. unsafe_allow_html=True yine gerekli (mevcut).
    """
    return "\n".join(s for s in (ln.strip() for ln in html.splitlines()) if s)


def render_map_metric_row(metrics_data: list, theme: dict) -> None:
    """Harita metrik satırı — CSS Grid ile garanti hizalama (height:110px sabit)."""
    cards_html = ""
    for m in metrics_data:
        label = m['label']
        value = m['value']
        color = m['color']
        filter_value = m.get('filter_value')
        key = m.get('key')

        inner = f"""
            <div style="text-align:center; height:100%;
                        display:flex; flex-direction:column;
                        justify-content:center; gap:0.4rem;">
                <div style="font-size:0.72rem; font-weight:600;
                            text-transform:uppercase; letter-spacing:0.05em;
                            color:{theme['text_secondary']};">
                    {label}
                </div>
                <div style="font-size:1.8rem; font-weight:700;
                            color:{color}; line-height:1;
                            letter-spacing:-0.03em;">
                    {value}
                </div>
            </div>
        """

        if filter_value and key:
            safe_val = ",".join(filter_value)
            cards_html += f"""
            <a href="?map_filter={safe_val}&map_key={key}"
               target="_self"
               style="text-decoration:none; color:inherit; display:block;">
                <div style="background:{theme['bg_tertiary']};
                            border-radius:12px;
                            padding:1rem;
                            box-shadow:0 1px 2px rgba(0,0,0,0.04);
                            border-left:4px solid {color};
                            height:110px;
                            box-sizing:border-box;
                            transition:all 0.2s;
                            cursor:pointer;"
                     onmouseover="this.style.transform='translateY(-2px)';"
                     onmouseout="this.style.transform='translateY(0)';">
                    {inner}
                </div>
            </a>
            """
        else:
            cards_html += f"""
            <div style="background:{theme['bg_tertiary']};
                        border-radius:12px;
                        padding:1rem;
                        box-shadow:0 1px 2px rgba(0,0,0,0.04);
                        border-left:4px solid {color};
                        height:110px;
                        box-sizing:border-box;">
                {inner}
            </div>
            """

    grid_html = f"""
    <div style="display:grid; grid-template-columns:repeat(4, 1fr);
                gap:1rem; margin-bottom:1rem;">
        {cards_html}
    </div>
    """
    st.markdown(_collapse_html(grid_html), unsafe_allow_html=True)


def build_popup_html(row: pd.Series) -> str:
    """Marker popup için HTML üret."""
    risk_color = {
        "critical": "#cf222e",
        "high": "#bc4c00",
        "medium": "#bf8700",
        "low": "#1a7f37",
    }.get(row['risk_level'], "#656d76")

    timestamp_str = pd.to_datetime(row['timestamp']).strftime('%d.%m.%Y %H:%M')

    thermal_block = ""
    if pd.notna(row.get('peak_temp')):
        thermal_block = f"""
        <div style="background:#fff8e1; padding:6px; border-radius:4px; margin-top:6px;
                    border-left:3px solid #bc4c00;">
            <strong>🌡️ Termal:</strong> {row['peak_temp']:.1f}°C peak
        </div>
        """

    return f"""
    <div style="font-family: 'Segoe UI', Arial, sans-serif; min-width: 240px;">
        <div style="background:{risk_color}; color:white; padding:8px 12px;
                    border-radius:4px 4px 0 0; font-weight:700; font-size:13px;">
            {row['risk_level_tr'].upper()} - Skor: {row['risk_score']:.1f}
        </div>
        <div style="padding:10px 12px; background:white;">
            <div style="margin-bottom:6px;">
                <strong>Arıza:</strong> {row['fault_type_label']}
            </div>
            <div style="margin-bottom:6px;">
                <strong>Ekipman:</strong> {row['equipment_label']}
            </div>
            <div style="margin-bottom:6px;">
                <strong>📍 Konum:</strong> {row['latitude']:.4f}°N, {row['longitude']:.4f}°E
            </div>
            <div style="margin-bottom:6px;">
                <strong>📅 Tarih:</strong> {timestamp_str}
            </div>
            <div style="margin-bottom:6px; color:#656d76; font-size:11px;">
                ID: {row['fault_id_short']}
            </div>
            {thermal_block}
            <div style="margin-top:10px; padding:8px;
                        background:#e3f2fd; border-radius:6px;
                        font-size:11px; color:#007aff; text-align:center;">
                🖱️ Marker'a <strong>tekrar tıkla</strong> → Detay sayfasına git
            </div>
        </div>
    </div>
    """


def create_map(df: pd.DataFrame) -> folium.Map:
    """Filtrelenmiş df'den Folium haritası oluştur (tema-bilinçli tile)."""
    theme = get_current_theme()
    default_tile = theme['map_tile']  # light → Positron, dark → Dark Matter

    if df.empty:
        # Boş harita - Türkiye merkezi (tema-bilinçli tile)
        return folium.Map(location=[39.0, 35.0], zoom_start=6,
                          tiles=default_tile)

    # Map merkezi: tüm noktaların ortalaması
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    # tiles=None → default'u istemiyoruz, tile'ları manuel ekliyoruz.
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles=None,
        control_scale=True,
    )

    # Default: aktif temaya göre Positron (açık) veya Dark Matter (koyu)
    folium.TileLayer(
        tiles=default_tile,
        name='Sade (Önerilen)',
        control=True,
    ).add_to(m)

    # Marker cluster
    cluster = MarkerCluster(name="Arızalar").add_to(m)

    for _, row in df.iterrows():
        color = FOLIUM_RISK_COLORS.get(row['risk_level'], 'gray')
        icon_name = {
            "critical": "exclamation-triangle",
            "high": "exclamation-circle",
            "medium": "info-circle",
            "low": "check-circle",
        }.get(row['risk_level'], 'circle')

        popup_html = build_popup_html(row)

        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=row['fault_id'],  # ⭐ fault_id buradan yakalanacak (click capture)
            icon=folium.Icon(color=color, icon=icon_name, prefix='fa'),
        ).add_to(cluster)

    # Alternatif tile'lar (default zaten eklendi; tekrarı atla)
    if default_tile != 'CartoDB positron':
        folium.TileLayer(tiles='CartoDB positron', name='Açık tema', control=True).add_to(m)
    if default_tile != 'CartoDB dark_matter':
        folium.TileLayer(tiles='CartoDB dark_matter', name='Koyu tema', control=True).add_to(m)
    folium.TileLayer(tiles='OpenStreetMap', name='Standart', control=True).add_to(m)

    # Layer control (sağ üstte tile değiştirme)
    folium.LayerControl(position='topright', collapsed=False).add_to(m)

    return m


def main() -> None:
    theme = get_current_theme()

    # QUERY PARAM ile harita risk filtre değişimi (doğrudan kart tıklamasından gelir)
    map_filter = st.query_params.get("map_filter", None)
    map_key = st.query_params.get("map_key", None)
    if map_filter and map_key:
        st.query_params.clear()  # tekrar tetiklenmesin diye temizle
        st.session_state['map_risk_filter'] = map_filter.split(",")
        st.rerun()

    # YENİ: Marker popup'taki "Detayda Gör" tıklaması → o arızayı Faults'ta aç
    focus_fault = st.query_params.get("focus_fault", None)
    if focus_fault:
        st.query_params.clear()
        st.session_state['faults_focus_id'] = focus_fault
        try:
            st.switch_page("pages/3_Faults.py")
        except Exception:
            # AppTest izole ortamında switch_page çalışmaz, ignore
            pass

    st.markdown('<h1 class="main-title">🗺️ Harita Görünümü</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Arızaların coğrafi dağılımı, risk renkli işaretler</p>',
                unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.warning("⚠️ GPS koordinatı olan arıza kaydı bulunamadı.")
        return

    # ==== FİLTRELER ====
    st.markdown("### 🔍 Filtreler")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        # Metrik kartlardan gelen tıklama filtresini uygula (tek seferlik)
        default_risk_map = st.session_state.get(
            'map_risk_filter', ['critical', 'high', 'medium', 'low'])
        if 'map_risk_filter' in st.session_state:
            del st.session_state['map_risk_filter']
        risk_filter = st.multiselect(
            "Risk Seviyesi",
            options=['critical', 'high', 'medium', 'low'],
            default=default_risk_map,
            format_func=lambda x: RISK_TR.get(x, x),
        )

    with fc2:
        all_flights = sorted(df['flight_id'].unique().tolist())
        flight_filter = st.selectbox(
            "Uçuş",
            options=['Hepsi'] + all_flights,
            format_func=lambda x: "Hepsi" if x == 'Hepsi' else FLIGHT_LABELS.get(x, x),
        )

    with fc3:
        all_types = sorted(df['fault_type'].unique().tolist())
        type_filter = st.multiselect(
            "Arıza Tipi",
            options=all_types,
            default=all_types,
            format_func=lambda x: FAULT_TYPE_LABELS.get(x, x),
        )

    # Filtreleri uygula
    filtered = df[df['risk_level'].isin(risk_filter) & df['fault_type'].isin(type_filter)]
    if flight_filter != 'Hepsi':
        filtered = filtered[filtered['flight_id'] == flight_filter]

    # ==== ÖZET METRİKLER ==== (CSS Grid — height:110px sabit, %100 hizalı)
    st.markdown("<br>", unsafe_allow_html=True)
    render_map_metric_row([
        {'label': 'GÖRÜNTÜLENEN', 'value': str(len(filtered)),
         'color': theme['accent_primary'],
         'filter_value': ['critical', 'high', 'medium', 'low'], 'key': 'all'},
        {'label': 'KRİTİK', 'value': f"{(filtered['risk_level'] == 'critical').sum()}",
         'color': theme['danger'], 'filter_value': ['critical'], 'key': 'critical'},
        {'label': 'YÜKSEK', 'value': f"{(filtered['risk_level'] == 'high').sum()}",
         'color': theme['orange'], 'filter_value': ['high'], 'key': 'high'},
        {'label': 'UÇUŞ', 'value': f"{filtered['flight_id'].nunique()}",
         'color': theme['accent_primary']},
    ], theme)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==== HARİTA ====
    if filtered.empty:
        st.warning("Seçili filtrelere uyan kayıt bulunmuyor.")
    else:
        m = create_map(filtered)
        # st_folium dönüş değeriyle marker tıklamasını yakala (sandbox link YOK)
        map_data = st_folium(
            m,
            returned_objects=['last_object_clicked_tooltip', 'last_object_clicked'],
            height=600,
            width=None,
            key="folium_map_main",
            use_container_width=True,
        )

        # Tıklama yakalama: last_object_clicked_tooltip = marker'a verdiğimiz fault_id
        if map_data:
            clicked_tooltip = map_data.get('last_object_clicked_tooltip')
            if clicked_tooltip and clicked_tooltip != st.session_state.get('_last_map_click'):
                st.session_state['faults_focus_id'] = clicked_tooltip
                st.session_state['_last_map_click'] = clicked_tooltip
                try:
                    st.switch_page("pages/3_Faults.py")
                except Exception:
                    # AppTest ortamında switch_page çalışmaz
                    st.info(f"Yönlendiriliyor: {clicked_tooltip[:8]}...")

    # ==== ALT TABLO (haritadaki noktaların listesi) ====
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📋 Görüntülenen Arızalar")

    if not filtered.empty:
        disp = filtered[['timestamp', 'flight_label', 'fault_type_label', 'equipment_label',
                         'risk_level_tr', 'risk_score', 'latitude', 'longitude']].copy()
        disp['timestamp'] = pd.to_datetime(disp['timestamp']).dt.strftime('%d.%m.%Y %H:%M')
        disp.columns = ['Tarih', 'Uçuş', 'Arıza Tipi', 'Ekipman', 'Risk', 'Skor', 'Enlem', 'Boylam']

        risk_bg = {
            'Kritik': f'background-color: {theme["danger"]}; color: white;',
            'Yüksek': f'background-color: {theme["orange"]}; color: white;',
            'Orta': f'background-color: {theme["warning"]}; color: white;',
            'Düşük': f'background-color: {theme["success"]}; color: white;',
        }

        styled = disp.style.format({
            'Skor': '{:.1f}',
            'Enlem': '{:.4f}',
            'Boylam': '{:.4f}',
        }).map(lambda v: risk_bg.get(v, ''), subset=['Risk'])

        st.dataframe(styled, width='stretch', hide_index=True)


main()
