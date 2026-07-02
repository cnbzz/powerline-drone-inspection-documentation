"""Dashboard summary page — KPIs, distribution charts, recent critical faults.

Reads exclusively from ``outputs/db/faults.db`` (no fabricated data).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime  # noqa: E402

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

# set_page_config must be the first Streamlit call.
st.set_page_config(page_title="Dashboard - HHATS", page_icon="📊", layout="wide")

from src.dashboard.app import (  # noqa: E402
    DB_PATH,
    get_current_theme,
    get_risk_colors,
    load_css,
    render_sidebar,
)
from src.dashboard.utils.excel_export import export_faults_to_excel  # noqa: E402
from src.fault.database import FaultDatabase  # noqa: E402
from src.thermal.thresholds import assess_thermal  # noqa: E402  # (kept for parity; counts use thermal_status)

load_css()

from src.auth.session import require_login  # noqa: E402

require_login()  # giriş kapısı — giriş yoksa login formu + st.stop()
render_sidebar()

STATUS_LABELS = {
    "new": "Yeni", "reviewed": "İncelendi", "resolved": "Çözüldü",
    "false_positive": "Yanlış Alarm",
}
FAULT_TYPE_LABELS = {
    "thermal_hotspot": "Termal Anomali", "visual_defect": "Görsel Hasar",
    "equipment_detected": "Normal Ekipman", "combined": "Çoklu Arıza",
    "vegetation_encroachment": "Bitki Örtüsü", "unknown": "Belirsiz",
}
RISK_TR = {"critical": "Kritik", "high": "Yüksek", "medium": "Orta", "low": "Düşük"}
FLIGHT_SHORT = {
    "flight_2026_05_01_kocaeli": "Kocaeli (01.05)",
    "flight_2026_05_08_sakarya": "Sakarya (08.05)",
    "flight_2026_05_15_bursa": "Bursa (15.05)",
    "flight_2026_05_22_istanbul": "İstanbul (22.05)",
}


@st.cache_data(ttl=60)
def load_data() -> pd.DataFrame:
    """Load every fault record into a DataFrame."""
    db = FaultDatabase(str(DB_PATH))
    rows = []
    for r in db.list_all(limit=10000):
        rows.append({
            "fault_id": r.fault_id,
            "timestamp": r.timestamp,
            "flight_id": r.inspection_flight_id or "—",
            "fault_type": r.fault_type.value,
            "fault_type_label": FAULT_TYPE_LABELS.get(r.fault_type.value, r.fault_type.value),
            "equipment_class": r.equipment_class or "—",
            "confidence": r.detection_confidence,
            "risk_level": r.risk_level.value,
            "risk_score": r.risk_score,
            "status": r.status.value,
            "status_label": STATUS_LABELS.get(r.status.value, r.status.value),
            "latitude": r.gps.latitude if r.gps else None,
            "longitude": r.gps.longitude if r.gps else None,
            "peak_temp": r.thermal.peak_temp_c if r.thermal else None,
            "delta_temp": r.thermal.temperature_delta_c if r.thermal else None,
            "thermal_status": (
                assess_thermal(r.thermal.peak_temp_c,
                               r.thermal.temperature_delta_c).final_status.value
                if r.thermal else None
            ),
        })
    return pd.DataFrame(rows)


def render_kpi_card(label: str, value: str, icon: str, color: str = "#007aff",
                    filter_key: str = None, filter_value=None) -> None:
    """KPI kartı — filter_key verilirse DOĞRUDAN tıklanabilir (buton YOK).

    Tıklamada ``?kpi_nav=<filter_key>&kpi_val=<value>`` query param'ı set edilir;
    main() bunu okuyup filter state'i atayıp Faults sayfasına yönlendirir.

    filter_key: hangi filtreyi ayarla ('faults_default_risk', 'faults_default_status',
                'faults_default_type')
    filter_value: filtre değeri (örn: ['critical'], ['new'], ['thermal_hotspot'])
    """
    if filter_key and filter_value is not None:
        safe_filter_value = (",".join(filter_value)
                             if isinstance(filter_value, list) else str(filter_value))
        st.markdown(f"""
        <a href="?kpi_nav={filter_key}&kpi_val={safe_filter_value}"
           target="_self" class="kpi-card-link">
            <div class="kpi-card-clickable" style="border-left: 4px solid {color};">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <div class="kpi-label">{label}</div>
                        <div class="kpi-value" style="color: {color};">{value}</div>
                    </div>
                    <div class="kpi-icon">{icon}</div>
                </div>
            </div>
        </a>
        """, unsafe_allow_html=True)
    else:
        # Tıklanamaz kart (sadece bilgi)
        st.markdown(f"""
        <div class="kpi-card" style="border-left: 4px solid {color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value" style="color: {color};">{value}</div>
                </div>
                <div class="kpi-icon">{icon}</div>
            </div>
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


def render_kpi_row(cards_data: list, theme: dict) -> None:
    """Tek HTML block'ta KPI satırı — Streamlit wrapper'sız.

    CSS Grid ile 4 sütun, kesinlikle eşit yükseklik (height:160px sabit).
    Streamlit'in stMarkdownContainer wrapper'ı tek kez sarar → hizalama garanti.

    cards_data: List[Dict] — her dict:
        - label, value, icon, color
        - filter_key (opsiyonel), filter_value (opsiyonel)
    """
    cards_html = ""
    for card in cards_data:
        label = card['label']
        value = card['value']
        icon = card['icon']
        color = card['color']
        filter_key = card.get('filter_key')
        filter_value = card.get('filter_value')

        inner_html = f"""
            <div style="display:flex; justify-content:space-between;
                        align-items:flex-start; height:100%; gap:0.5rem;">
                <div style="flex:1; min-width:0;">
                    <div style="font-size:0.72rem; font-weight:600;
                                text-transform:uppercase; letter-spacing:0.05em;
                                color:{theme['text_secondary']};
                                line-height:1.3;
                                white-space:normal;
                                word-wrap:break-word;
                                margin-bottom:0.5rem;
                                min-height:2.6em;">
                        {label}
                    </div>
                    <div style="font-size:2.25rem; font-weight:700;
                                color:{color}; line-height:1;
                                letter-spacing:-0.03em;
                                margin-top:0.25rem;">
                        {value}
                    </div>
                </div>
                <div style="font-size:1.5rem; opacity:0.8; flex-shrink:0;">
                    {icon}
                </div>
            </div>
        """

        if filter_key and filter_value is not None:
            safe_val = (",".join(filter_value)
                        if isinstance(filter_value, list) else str(filter_value))
            cards_html += f"""
            <a href="?kpi_nav={filter_key}&kpi_val={safe_val}"
               target="_self"
               style="text-decoration:none; color:inherit; display:block;">
                <div style="background:{theme['bg_tertiary']};
                            border-radius:16px;
                            padding:1.25rem 1.5rem;
                            box-shadow:0 1px 2px rgba(0,0,0,0.04);
                            border-left:4px solid {color};
                            height:160px;
                            box-sizing:border-box;
                            transition:all 0.2s;
                            cursor:pointer;"
                     onmouseover="this.style.transform='translateY(-3px)';
                                  this.style.boxShadow='0 8px 24px rgba(0,0,0,0.1)';"
                     onmouseout="this.style.transform='translateY(0)';
                                 this.style.boxShadow='0 1px 2px rgba(0,0,0,0.04)';">
                    {inner_html}
                </div>
            </a>
            """
        else:
            cards_html += f"""
            <div style="background:{theme['bg_tertiary']};
                        border-radius:16px;
                        padding:1.25rem 1.5rem;
                        box-shadow:0 1px 2px rgba(0,0,0,0.04);
                        border-left:4px solid {color};
                        height:160px;
                        box-sizing:border-box;">
                {inner_html}
            </div>
            """

    grid_html = f"""
    <div style="display:grid;
                grid-template-columns:repeat(4, 1fr);
                gap:1rem;
                margin-bottom:1rem;">
        {cards_html}
    </div>
    """
    st.markdown(_collapse_html(grid_html), unsafe_allow_html=True)


def main() -> None:
    theme = get_current_theme()
    RISK_COLORS = get_risk_colors()  # theme-aware, shadows module import

    # QUERY PARAM ile KPI yönlendirme (doğrudan kart tıklamasından gelir)
    kpi_nav = st.query_params.get("kpi_nav", None)
    kpi_val = st.query_params.get("kpi_val", None)
    if kpi_nav and kpi_val:
        st.query_params.clear()  # tekrar tetiklenmesin diye temizle
        st.session_state[kpi_nav] = kpi_val.split(",")
        try:
            st.switch_page("pages/3_Faults.py")
        except Exception:
            # AppTest izole ortamında switch_page çalışmaz, ignore
            pass

    st.markdown('<h1 class="main-title">📊 Genel Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Sistem durumu, performans göstergeleri ve risk dağılımı</p>',
                unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.warning("⚠️ Veritabanında kayıt bulunamadı. Önce `generate_demo_data.py` çalıştırın.")
        return

    thermal_count = df['fault_type'].isin(['thermal_hotspot', 'combined']).sum()

    # KPI satırı 1 — tek HTML block, CSS Grid (height:160px sabit, %100 hizalı)
    render_kpi_row([
        {'label': 'TOPLAM ARIZA', 'value': f"{len(df)}", 'icon': '🔍',
         'color': theme['accent_primary']},
        {'label': 'KRİTİK RİSK', 'value': f"{(df['risk_level'] == 'critical').sum()}",
         'icon': '🔴', 'color': theme['danger'],
         'filter_key': 'faults_default_risk', 'filter_value': ['critical']},
        {'label': 'YÜKSEK RİSK', 'value': f"{(df['risk_level'] == 'high').sum()}",
         'icon': '🟠', 'color': theme['orange'],
         'filter_key': 'faults_default_risk', 'filter_value': ['high']},
        {'label': 'TOPLAM UÇUŞ', 'value': f"{df['flight_id'].nunique()}", 'icon': '✈️',
         'color': theme['accent_primary']},
    ], theme)

    # KPI satırı 2
    render_kpi_row([
        {'label': 'YENİ İNCELEME', 'value': f"{(df['status'] == 'new').sum()}",
         'icon': '📝', 'color': theme['accent_primary'],
         'filter_key': 'faults_default_status', 'filter_value': ['new']},
        {'label': 'ÇÖZÜLDÜ', 'value': f"{(df['status'] == 'resolved').sum()}",
         'icon': '✅', 'color': theme['success'],
         'filter_key': 'faults_default_status', 'filter_value': ['resolved']},
        {'label': 'ORT. RİSK SKORU', 'value': f"{df['risk_score'].mean():.1f}",
         'icon': '📈', 'color': theme['accent_primary']},
        {'label': 'TERMAL ANOMALİ', 'value': f"{thermal_count}",
         'icon': '🌡️', 'color': theme['danger'],
         'filter_key': 'faults_default_type', 'filter_value': ['thermal_hotspot']},
    ], theme)

    # KPI satırı 3 — Termal durum (sabit endüstri eşikleri 60/80°C, ΔT 10/30°C)
    ts = df['thermal_status']
    thermal_normal_count = int((ts == 'normal').sum())
    thermal_warning_count = int((ts == 'warning').sum())
    thermal_critical_count = int((ts == 'critical').sum())
    peak_mean = (f"{df['peak_temp'].mean():.1f}°C"
                 if df['peak_temp'].notna().any() else '—')
    render_kpi_row([
        {'label': 'TERMAL DİKKAT', 'value': str(thermal_warning_count),
         'icon': '🟡', 'color': theme['warning'],
         'filter_key': 'faults_default_thermal', 'filter_value': ['warning']},
        {'label': 'TERMAL KRİTİK', 'value': str(thermal_critical_count),
         'icon': '🔴', 'color': theme['danger'],
         'filter_key': 'faults_default_thermal', 'filter_value': ['critical']},
        {'label': 'TERMAL NORMAL', 'value': str(thermal_normal_count),
         'icon': '🟢', 'color': theme['success']},
        {'label': 'ORT. PEAK SICAKLIK', 'value': peak_mean,
         'icon': '🌡️', 'color': theme['accent_primary']},
    ], theme)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Distribution charts
    st.markdown("### 📊 Dağılım Analizleri")
    col1, col2 = st.columns(2)
    with col1:
        risk_counts = df['risk_level'].value_counts()
        order = ['critical', 'high', 'medium', 'low']
        fig = go.Figure(data=[go.Pie(
            labels=[RISK_TR[r] for r in order],
            values=[int(risk_counts.get(r, 0)) for r in order],
            marker=dict(colors=[RISK_COLORS[r] for r in order]),
            hole=0.45, textinfo='label+percent', textfont=dict(size=12, color='white'),
        )])
        fig.update_layout(title="Risk Seviyesi Dağılımı", title_font=dict(size=18, color=theme['text_primary']),
                          height=350, margin=dict(t=50, b=20, l=20, r=20),
                          paper_bgcolor=theme['chart_paper'], plot_bgcolor=theme['chart_plot'],
                          font=dict(color=theme['text_primary']))
        # Tıklanabilir — dilime tıklayınca o risk seviyesi filtreli Faults'a git
        event = st.plotly_chart(fig, width='stretch', on_select='rerun',
                                selection_mode='points', key='risk_pie_chart')
        if event and event.get('selection', {}).get('points'):
            clicked_label = event['selection']['points'][0].get('label', '').lower()
            risk_map = {'kritik': 'critical', 'yüksek': 'high', 'orta': 'medium', 'düşük': 'low'}
            if clicked_label in risk_map:
                st.session_state['faults_default_risk'] = [risk_map[clicked_label]]
                st.switch_page("pages/3_Faults.py")
    with col2:
        status_counts = df['status_label'].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=status_counts.values, y=status_counts.index, orientation='h',
            marker=dict(color=[theme['accent_primary'], theme['success'], theme['warning'],
                               theme['text_secondary']][:len(status_counts)]),
            text=status_counts.values, textposition='outside',
        )])
        fig.update_layout(title="Arıza Durumu", title_font=dict(size=18, color=theme['text_primary']),
                          xaxis_title="Sayı", height=350, margin=dict(t=50, b=20, l=20, r=40),
                          paper_bgcolor=theme['chart_paper'], plot_bgcolor=theme['chart_plot'],
                          font=dict(color=theme['text_primary']))
        st.plotly_chart(fig, width='stretch')

    col1, col2 = st.columns(2)
    with col1:
        flight_risk = df.groupby(['flight_id', 'risk_level']).size().unstack(fill_value=0)
        flight_risk = flight_risk.reindex(columns=['critical', 'high', 'medium', 'low'], fill_value=0)
        flight_risk.index = [FLIGHT_SHORT.get(idx, idx) for idx in flight_risk.index]
        fig = go.Figure()
        for risk in ['critical', 'high', 'medium', 'low']:
            fig.add_trace(go.Bar(name=RISK_TR[risk], x=flight_risk.index, y=flight_risk[risk],
                                 marker_color=RISK_COLORS[risk]))
        fig.update_layout(title="Uçuş Bazında Risk Dağılımı", title_font=dict(size=18, color=theme['text_primary']),
                          barmode='stack', yaxis_title="Arıza Sayısı", height=350,
                          margin=dict(t=50, b=20, l=20, r=20), legend=dict(orientation='h', y=-0.15),
                          paper_bgcolor=theme['chart_paper'], plot_bgcolor=theme['chart_plot'],
                          font=dict(color=theme['text_primary']))
        # Tıklanabilir — bara tıklayınca o uçuş filtreli Faults'a git
        event = st.plotly_chart(fig, width='stretch', on_select='rerun',
                                selection_mode='points', key='flight_bar_chart')
        if event and event.get('selection', {}).get('points'):
            clicked_flight = event['selection']['points'][0].get('x', '')
            for fid, short in FLIGHT_SHORT.items():
                if clicked_flight in short or short.split(' (')[0] == clicked_flight:
                    st.session_state['faults_default_flight'] = fid
                    st.switch_page("pages/3_Faults.py")
                    break
    with col2:
        type_counts = df['fault_type_label'].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=type_counts.index, values=type_counts.values, hole=0.45,
            textinfo='label+percent',
            marker=dict(colors=[theme['accent_primary'], theme['danger'], theme['warning'],
                                theme['success'], theme['text_secondary'], theme['orange']]),
        )])
        fig.update_layout(title="Arıza Tipi Dağılımı", title_font=dict(size=18, color=theme['text_primary']),
                          height=350, margin=dict(t=50, b=20, l=20, r=20),
                          paper_bgcolor=theme['chart_paper'], plot_bgcolor=theme['chart_plot'],
                          font=dict(color=theme['text_primary']))
        st.plotly_chart(fig, width='stretch')

    # Recent critical faults
    st.markdown("### 🚨 Son Kritik Arızalar")
    crit = df[df['risk_level'].isin(['critical', 'high'])].sort_values('risk_score', ascending=False).head(10)
    if not crit.empty:
        crit = crit.reset_index(drop=True)  # iloc[selected_idx] satır seçimi için
        disp = crit[['fault_id', 'timestamp', 'flight_id', 'fault_type_label', 'equipment_class',
                     'risk_level', 'risk_score', 'status_label']].copy()
        disp['timestamp'] = pd.to_datetime(disp['timestamp']).dt.strftime('%d.%m.%Y %H:%M')
        disp['flight_id'] = disp['flight_id'].map(lambda x: FLIGHT_SHORT.get(x, x).split(' (')[0])
        disp['risk_level'] = disp['risk_level'].map(RISK_TR)
        # _fault_id görünmeyecek (selection için saklanır)
        disp.columns = ['_fault_id', 'Tarih', 'Uçuş', 'Arıza Tipi', 'Ekipman', 'Risk', 'Skor', 'Durum']

        risk_bg = {
            'Kritik': f'background-color: {theme["danger"]}; color: white;',
            'Yüksek': f'background-color: {theme["orange"]}; color: white;',
            'Orta': f'background-color: {theme["warning"]}; color: white;',
            'Düşük': f'background-color: {theme["success"]}; color: white;',
        }
        styled = (disp.drop(columns=['_fault_id'])
                      .style.format({'Skor': '{:.1f}'})
                      .map(lambda v: risk_bg.get(v, ''), subset=['Risk']))

        # Satır seçilince o arızayı Faults sayfasında aç
        st.caption("💡 Bir satıra tıklayarak ilgili arızanın detayına gidebilirsiniz.")
        event = st.dataframe(
            styled, width='stretch', hide_index=True,
            on_select='rerun', selection_mode='single-row',
            key='dash_critical_table',
        )
        if event.selection and event.selection.get('rows'):
            selected_idx = event.selection['rows'][0]
            st.session_state['faults_focus_id'] = disp.iloc[selected_idx]['_fault_id']
            st.switch_page("pages/3_Faults.py")

        # Yönlendirme + tüm-veri Excel export
        st.markdown("<br>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("🔍 Tüm Kritik Arızaları Görüntüle →",
                         key="goto_faults_critical", width='stretch', type='primary'):
                # Faults sayfasına yönlendir + filtre state'i ayarla (tek seferlik)
                st.session_state['faults_default_risk'] = ['critical']
                st.switch_page("pages/3_Faults.py")
        with bcol2:
            # Tüm kayıtları (sadece kritik değil) Excel'e dök — export şemasına uyarla
            export_df = df.copy()
            export_df['fault_id_short'] = export_df['fault_id'].str[:8]
            export_df['flight_label'] = export_df['flight_id'].map(lambda x: FLIGHT_SHORT.get(x, x))
            export_df['equipment_label'] = export_df['equipment_class']
            export_df['risk_level_tr'] = export_df['risk_level'].map(RISK_TR)
            export_df['status_tr'] = export_df['status_label']
            excel_bytes_full = export_faults_to_excel(export_df)
            st.download_button(
                label="📥 Tüm Verileri Excel Olarak İndir",
                data=excel_bytes_full,
                file_name=f"hhats_tum_arizalar_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dash_excel_full",
                width='stretch',
            )
    else:
        st.info("Kritik veya yüksek riskli arıza bulunmuyor.")

    # Time series
    st.markdown("### 📈 Zaman İçinde Arıza Tespitleri")
    df = df.copy()
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    ts = df.groupby(['date', 'risk_level']).size().unstack(fill_value=0)
    ts = ts.reindex(columns=['critical', 'high', 'medium', 'low'], fill_value=0)
    fig = go.Figure()
    for risk in ['low', 'medium', 'high', 'critical']:
        fig.add_trace(go.Scatter(
            x=list(ts.index), y=ts[risk], name=RISK_TR[risk], mode='lines+markers',
            stackgroup='one', line=dict(color=RISK_COLORS[risk], width=2), marker=dict(size=8),
        ))
    fig.update_layout(title="Günlük Arıza Tespitleri (Risk Bazında)",
                      title_font=dict(size=18, color=theme['text_primary']),
                      xaxis_title="Tarih", yaxis_title="Tespit Sayısı", height=400,
                      margin=dict(t=50, b=20, l=20, r=20), hovermode='x unified',
                      paper_bgcolor=theme['chart_paper'], plot_bgcolor=theme['chart_plot'],
                      font=dict(color=theme['text_primary']))
    st.plotly_chart(fig, width='stretch')


main()
