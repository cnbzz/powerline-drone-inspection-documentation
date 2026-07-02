"""HHATS dashboard entry point (Streamlit multipage).

Run with::

    streamlit run src/dashboard/app.py

Pages under ``src/dashboard/pages/`` are auto-discovered. This module hosts the
landing page and exposes :func:`load_css` / shared constants for the pages.

Note: ``st.set_page_config`` is called inside :func:`main` (not at import time) so
pages can safely ``from src.dashboard.app import load_css`` without triggering a
duplicate page-config error.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap project root onto sys.path so ``src...`` imports resolve when Streamlit
# runs this file directly (sys.path[0] would otherwise be src/dashboard).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import base64  # noqa: E402

import streamlit as st  # noqa: E402

DB_PATH = _ROOT / "outputs" / "db" / "faults.db"

# Canonical BEDAŞ logo path. White-background PNG (322×156, ~2.06:1) — surfaces
# place it inside a light rounded container so it stays visible on dark themes.
LOGO_PATH = _ROOT / "assets" / "logo" / "bedas_logo.png"


@st.cache_data
def _logo_base64() -> str | None:
    """Return the BEDAŞ logo as a base64 string (cached), or ``None`` if the file
    is missing. Callers embed it via an ``<img>`` tag so they keep full control of
    sizing/centering; aspect ratio is preserved by setting width only."""
    try:
        if not LOGO_PATH.exists():
            return None
        return base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except Exception:
        return None


def logo_img_html(width_px: int, *, container: bool = True,
                  margin: str = "0 auto 0.75rem auto",
                  align: str = "center") -> str | None:
    """Build an ``<img>`` HTML snippet for the logo at ``width_px`` (height auto →
    aspect ratio preserved), aligned per ``align`` (``center``/``left``). When
    ``container`` is True the logo sits on a white rounded card so it reads on dark
    backgrounds. Returns ``None`` if the logo file is missing (caller falls back to
    text/emoji)."""
    b64 = _logo_base64()
    if not b64:
        return None
    img = (
        f"<img src='data:image/png;base64,{b64}' "
        f"style='width:{width_px}px; height:auto; display:block;' "
        f"alt='BEDAŞ' />"
    )
    if not container:
        inner = img
    else:
        # White rounded card so the white-bg logo reads on dark themes too.
        inner = (
            f"<span style='background:#ffffff; border-radius:14px; "
            f"padding:12px 18px; display:inline-block; "
            f"box-shadow:0 1px 4px rgba(0,0,0,0.08);'>{img}</span>"
        )
    # Outer block aligns the (inline-block) card via text-align.
    return f"<div style='text-align:{align}; margin:{margin};'>{inner}</div>"

# --------------------------------------------------------------------------- #
# Theme system — two palettes, switchable via sidebar toggle, persisted in     #
# st.session_state["theme"] (survives page navigation automatically).          #
# --------------------------------------------------------------------------- #
THEMES = {
    "light": {
        "name": "light",
        "icon": "☀️",
        "label": "Açık",
        # Apple Light Mode - iOS Grouped Style
        "bg_primary": "#f2f2f7",        # iOS Grouped Background
        "bg_secondary": "#ffffff",       # Card/Sidebar (Apple uses pure white)
        "bg_tertiary": "#ffffff",        # Inner cards (same as secondary)
        "border_light": "#e5e5ea",       # iOS Separator
        "border_medium": "#d1d1d6",      # iOS Separator dark
        "accent_primary": "#007aff",     # iOS Blue
        "accent_secondary": "#ff3b30",   # iOS Red
        "text_primary": "#1c1c1e",       # iOS Label
        "text_secondary": "#8e8e93",     # iOS Secondary Label
        "text_muted": "#c7c7cc",         # iOS Tertiary Label
        # iOS System Colors (slightly muted)
        "success": "#34c759",            # iOS Green
        "warning": "#ffcc00",            # iOS Yellow
        "orange": "#ff9500",             # iOS Orange
        "danger": "#ff3b30",             # iOS Red
        # Plotly chart bg
        "chart_paper": "#ffffff",
        "chart_plot": "#ffffff",
        # Folium tile
        "map_tile": "CartoDB positron",
    },
    "dark": {
        "name": "dark",
        "icon": "🌙",
        "label": "Koyu",
        # Apple Dark Mode - iOS Pure Black
        "bg_primary": "#000000",         # iOS Dark Background
        "bg_secondary": "#1c1c1e",       # iOS Secondary Grouped
        "bg_tertiary": "#2c2c2e",        # iOS Tertiary
        "border_light": "#38383a",       # iOS Dark Separator
        "border_medium": "#48484a",
        "accent_primary": "#0a84ff",     # iOS Dark Mode Blue
        "accent_secondary": "#ff453a",   # iOS Dark Mode Red
        "text_primary": "#ffffff",       # iOS Label Dark
        "text_secondary": "#98989f",     # iOS Secondary Label Dark
        "text_muted": "#48484a",
        # iOS Dark System Colors
        "success": "#30d158",
        "warning": "#ffd60a",
        "orange": "#ff9f0a",
        "danger": "#ff453a",
        "chart_paper": "#1c1c1e",
        "chart_plot": "#000000",
        "map_tile": "CartoDB dark_matter",
    },
}


def init_theme() -> None:
    """Initialize theme in session state if not set."""
    if "theme" not in st.session_state:
        st.session_state.theme = "light"


def get_current_theme() -> dict:
    """Return active theme palette."""
    init_theme()
    return THEMES[st.session_state.theme]


def toggle_theme() -> None:
    """Toggle between light and dark."""
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"


def get_risk_colors() -> dict:
    """Theme-aware risk colors for charts/markers."""
    theme = get_current_theme()
    return {
        "low": theme["success"],
        "medium": theme["warning"],
        "high": theme["orange"],
        "critical": theme["danger"],
    }


# Backward-compatible static risk colors (light palette). New standard:
# get_risk_colors() for theme-aware charts/markers.
RISK_COLORS = {
    "low": "#1a7f37",
    "medium": "#bf8700",
    "high": "#bc4c00",
    "critical": "#cf222e",
}


@st.cache_data
def _build_css(theme_name: str) -> str:
    """Build the theme CSS string. Cached by ``theme_name`` (deterministic key) so
    the f-string interpolation only runs once per theme, not on every page load."""
    theme = THEMES[theme_name]

    css = f"""
    <style>
    :root {{
        --bg-primary: {theme['bg_primary']};
        --bg-secondary: {theme['bg_secondary']};
        --bg-tertiary: {theme['bg_tertiary']};
        --border-light: {theme['border_light']};
        --border-medium: {theme['border_medium']};
        --accent-primary: {theme['accent_primary']};
        --accent-secondary: {theme['accent_secondary']};
        --text-primary: {theme['text_primary']};
        --text-secondary: {theme['text_secondary']};
        --text-muted: {theme['text_muted']};
        --success: {theme['success']};
        --warning: {theme['warning']};
        --orange: {theme['orange']};
        --danger: {theme['danger']};
    }}

    /* Ana container */
    .main, .stApp {{
        background-color: {theme['bg_primary']} !important;
    }}

    /* Başlık - tema bağımsız (literal renk + !important, override yok) */
    .main-title {{
        color: {theme['text_primary']} !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-bottom: none;  /* Apple çizgi yerine spacing kullanır */
        padding-bottom: 0;
        letter-spacing: -0.03em;  /* Apple tight tracking */
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display",
                     "Segoe UI", Roboto, sans-serif;
    }}

    /* Tüm h1-h6 başlıkları tema rengini al */
    h1, h2, h3, h4, h5, h6 {{
        color: {theme['text_primary']} !important;
    }}

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: {theme['text_primary']} !important;
    }}

    .subtitle {{
        color: {theme['text_secondary']};
        font-size: 0.95rem;
        margin-bottom: 2.5rem;  /* daha fazla breathing room */
        font-weight: 400;
    }}

    /* Body text + global Apple system font */
    body, .stApp, .stMarkdown, p, span, div {{
        color: {theme['text_primary']};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", Roboto, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    /* KPI Kartları — Apple style (yumuşak köşe, subtle shadow, eşit yükseklik) */
    .kpi-card {{
        background: {theme['bg_tertiary']};
        border-radius: 16px;  /* Apple-style large radius */
        padding: 1.5rem 1.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 0 1px rgba(0,0,0,0.06);
        border: none;  /* No border, just shadow */
        height: 100%;          /* st.columns parent yüksekliğini doldur */
        min-height: 130px;     /* tüm kartlar minimum aynı yükseklik */
        display: flex;
        flex-direction: column;
        justify-content: space-between;  /* içeriği üst+alt yay */
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;  /* Tıklanabilir hissi */
    }}

    .kpi-card:hover {{
        box-shadow: 0 4px 16px rgba(0,0,0,0.08), 0 1px 4px rgba(0,0,0,0.04);
        transform: translateY(-2px);
    }}

    /* st.columns — eşit yükseklik garanti */
    [data-testid="stHorizontalBlock"] {{
        align-items: stretch !important;
    }}

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        display: flex;
        flex-direction: column;
    }}

    .kpi-label {{
        color: {theme['text_secondary']};
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    /* Renk vurgusu — kenarlık yerine value text rengi */
    .kpi-value {{
        color: {theme['text_primary']};
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
        letter-spacing: -0.03em;
    }}

    .kpi-icon {{
        font-size: 2rem;
        opacity: 0.5;
    }}

    /* === DOĞRUDAN TIKLANABILIR NAV CARDS (Buton YOK) === */
    /* Link decoration kaldır - kart tıklaması link gibi davranmasın */
    .nav-card-link {{
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        margin-bottom: 1.25rem;
    }}

    .nav-card-link:hover {{
        text-decoration: none !important;
    }}

    /* Apple-style direkt tıklanabilir kart */
    .nav-card-direct {{
        background: {theme['bg_tertiary']};
        border-radius: 16px;
        padding: 1.75rem 2rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 0 1px rgba(0,0,0,0.06);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        min-height: 130px;
        cursor: pointer;
        user-select: none;
    }}

    .nav-card-direct:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.05);
        transform: translateY(-4px);
        background: {theme['bg_secondary']};
    }}

    .nav-card-direct:active {{
        transform: translateY(-1px);
        transition: all 0.1s;
    }}

    /* === KPI CARDS DOĞRUDAN TIKLANABILIR === */
    .kpi-card-link {{
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        height: 100%;
    }}

    .kpi-card-link:hover {{
        text-decoration: none !important;
    }}

    .kpi-card-clickable {{
        background: {theme['bg_tertiary']};
        border-radius: 16px;
        padding: 1.5rem 1.75rem;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04), 0 0 1px rgba(0,0,0,0.06);
        height: 100%;
        min-height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        user-select: none;
    }}

    .kpi-card-clickable:hover {{
        box-shadow: 0 8px 24px rgba(0,0,0,0.1), 0 2px 6px rgba(0,0,0,0.05);
        transform: translateY(-3px);
        background: {theme['bg_secondary']};
    }}

    .kpi-card-clickable:active {{
        transform: translateY(-1px);
        transition: all 0.1s;
    }}

    /* Risk Badges */
    .badge-low {{ background: {theme['success']}; color: white; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; }}
    .badge-medium {{ background: {theme['warning']}; color: white; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; }}
    .badge-high {{ background: {theme['orange']}; color: white; padding: 4px 12px; border-radius: 6px; font-size: 0.85rem; font-weight: 600; }}
    .badge-critical {{
        background: {theme['danger']}; color: white; padding: 4px 12px; border-radius: 6px;
        font-size: 0.85rem; font-weight: 600; animation: pulse 2s infinite;
    }}
    @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.7; }} }}

    /* SİDEBAR */
    section[data-testid="stSidebar"] {{
        background: {theme['bg_secondary']} !important;
        border-right: 1px solid {theme['border_light']};
    }}

    section[data-testid="stSidebar"] * {{
        color: {theme['text_primary']} !important;
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {theme['text_primary']} !important;
    }}

    section[data-testid="stSidebar"] h1 {{
        font-size: 1.4rem;
        border-bottom: 2px solid {theme['accent_secondary']};
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }}

    /* Sidebar navigation links */
    section[data-testid="stSidebar"] a {{
        color: {theme['text_secondary']} !important;
        font-weight: 500;
    }}

    section[data-testid="stSidebar"] a:hover {{
        color: {theme['accent_primary']} !important;
    }}

    /* Butonlar — iOS style (rounded, no shadow) */
    .stButton button {{
        background: {theme['accent_primary']};
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s;
        box-shadow: none;
    }}

    .stButton button:hover {{
        background: {theme['accent_primary']};
        opacity: 0.9;
        color: white !important;
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.25);
    }}

    .stButton button:active {{
        transform: scale(0.98);
    }}

    /* Tablo stilleri */
    .stDataFrame, .dataframe {{
        background: {theme['bg_tertiary']} !important;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {theme['border_light']};
    }}

    .dataframe thead th {{
        background: {theme['bg_secondary']} !important;
        color: {theme['text_primary']} !important;
        font-weight: 600 !important;
        border-bottom: 1px solid {theme['border_medium']} !important;
    }}

    .dataframe tbody td {{
        color: {theme['text_primary']} !important;
        background: {theme['bg_tertiary']} !important;
        border-bottom: 1px solid {theme['border_light']} !important;
    }}

    /* === ST.DATAFRAME — TEMA UYUMLU (dark mode fix) === */
    .stDataFrame, [data-testid="stDataFrame"] {{
        background: {theme['bg_tertiary']} !important;
        border-radius: 12px;
        border: 1px solid {theme['border_light']};
    }}

    /* Tablo iç container */
    [data-testid="stDataFrame"] > div {{
        background: {theme['bg_tertiary']} !important;
    }}

    /* Tablo header satırı */
    [data-testid="stDataFrame"] thead tr th {{
        background: {theme['bg_primary']} !important;
        color: {theme['text_primary']} !important;
        font-weight: 600 !important;
        border-bottom: 2px solid {theme['border_medium']} !important;
    }}

    /* Tablo body hücreleri */
    [data-testid="stDataFrame"] tbody tr td {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        border-bottom: 1px solid {theme['border_light']} !important;
    }}

    /* Hover satır */
    [data-testid="stDataFrame"] tbody tr:hover td {{
        background: {theme['bg_secondary']} !important;
    }}

    /* Seçili satır */
    [data-testid="stDataFrame"] tbody tr[data-state="selected"] td {{
        background: {theme['accent_primary']}20 !important;
    }}

    /* Faults pure-HTML tablo: tıklanabilir satır hover (onmouseover sanitize edilir,
       bu yüzden CSS :hover kullanılır) */
    tbody tr.fault-row:hover {{
        background: {theme['accent_primary']}20 !important;
        transition: background 0.15s;
    }}

    /* Streamlit info/warning kutuları */
    .stAlert {{
        border-radius: 6px;
        border: 1px solid {theme['border_light']};
        background: {theme['bg_tertiary']};
    }}

    /* Streamlit default branding gizle */
    /* Streamlit native header — collapse butonu gizlenmemeli, header görünür kalır */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Plotly chart container */
    .js-plotly-plot {{
        background: {theme['bg_tertiary']} !important;
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid {theme['border_light']};
    }}

    /* Streamlit input widgets (selectbox, multiselect, vb.) */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border_light']} !important;
    }}

    /* Selectbox dropdown items */
    .stSelectbox [data-baseweb="select"] *,
    .stMultiSelect [data-baseweb="select"] * {{
        color: {theme['text_primary']} !important;
    }}

    /* === MULTISELECT TAG DÜZELTMESİ === */
    /* Streamlit multiselect seçili öğelerinin BaseWeb internal stilini override */
    .stMultiSelect [data-baseweb="tag"] {{
        background-color: {theme['accent_primary']} !important;
        color: white !important;
        border: none !important;
    }}

    .stMultiSelect [data-baseweb="tag"] span {{
        color: white !important;
    }}

    .stMultiSelect [data-baseweb="tag"] svg {{
        fill: white !important;
    }}

    /* Multiselect dropdown menü öğeleri */
    [data-baseweb="popover"] li {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
    }}

    [data-baseweb="popover"] li:hover {{
        background: {theme['bg_secondary']} !important;
    }}

    /* Text input + selectbox iç metin */
    .stTextInput input, .stSelectbox > div > div {{
        color: {theme['text_primary']} !important;
    }}

    /* Date input */
    .stDateInput > div > div > input {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
    }}

    /* === EXPANDER (st.expander) DÜZELTMESİ === */
    /* Light tema'da expander koyu kalmasın */
    [data-testid="stExpander"] {{
        background: {theme['bg_tertiary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 8px !important;
    }}

    [data-testid="stExpander"] > details {{
        background: {theme['bg_tertiary']} !important;
    }}

    [data-testid="stExpander"] summary {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
    }}

    [data-testid="stExpander"] summary:hover {{
        background: {theme['bg_secondary']} !important;
    }}

    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        padding: 14px !important;
    }}

    /* Expander içindeki tüm yazılar ana metin rengini al */
    [data-testid="stExpander"] * {{
        color: {theme['text_primary']};
    }}

    /* === ST.TEXT için metin rengi düzeltme === */
    /* st.text widget'ı bazı tema bg'lerinde okunmuyor */
    .stText, [data-testid="stText"] {{
        color: {theme['text_primary']} !important;
    }}

    /* === MULTISELECT INPUT KUTUSU (TAM BEYAZ TEMA) === */
    /* Streamlit multiselect'in MAIN INPUT alanı BaseWeb default'u ile koyu kalıyor */
    .stMultiSelect > div {{
        background: {theme['bg_tertiary']} !important;
    }}

    .stMultiSelect > div > div {{
        background: {theme['bg_tertiary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
        min-height: 42px !important;
    }}

    .stMultiSelect > div > div:focus-within {{
        border-color: {theme['accent_primary']} !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15) !important;
    }}

    /* MultiSelect içindeki input alanı (yazma alanı) */
    .stMultiSelect [data-baseweb="select"] > div {{
        background: {theme['bg_tertiary']} !important;
    }}

    /* Selectbox de aynı stil */
    .stSelectbox > div > div {{
        background: {theme['bg_tertiary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
        min-height: 42px !important;
    }}

    .stSelectbox > div > div:focus-within {{
        border-color: {theme['accent_primary']} !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.15) !important;
    }}

    /* Text input + textarea */
    .stTextInput > div > div {{
        background: {theme['bg_tertiary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
    }}

    .stTextInput input {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
    }}

    .stTextArea textarea {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
    }}

    /* Number input */
    .stNumberInput > div > div {{
        background: {theme['bg_tertiary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
    }}

    .stNumberInput input {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
    }}

    /* === DOWNLOAD BUTTON — APPLE SECONDARY === */
    .stDownloadButton button {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['accent_primary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
        padding: 6px 16px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
    }}

    .stDownloadButton button:hover {{
        background: {theme['bg_secondary']} !important;
        border-color: {theme['accent_primary']} !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}

    /* === STREAMLIT NATIVE TOGGLE → iOS STYLE === */
    [data-testid="stToggle"] > label > div[role="checkbox"] {{
        background: {theme['border_medium']} !important;
        border-radius: 31px !important;
    }}

    [data-testid="stToggle"] > label > div[role="checkbox"][aria-checked="true"] {{
        background: {theme['accent_primary']} !important;
    }}

    /* === SIDEBAR BUTON STİLİ (Hızlı Durum tıklanabilir kartları) === */
    [data-testid="stSidebar"] .stButton button {{
        background: {theme['bg_tertiary']} !important;
        color: {theme['text_primary']} !important;
        border: 1px solid {theme['border_light']} !important;
        text-align: left !important;
        padding: 0.5rem 0.75rem !important;
        font-weight: 500 !important;
    }}

    [data-testid="stSidebar"] .stButton button:hover {{
        background: {theme['accent_primary']}15 !important;
        border-color: {theme['accent_primary']} !important;
    }}

    /* === SIDEBAR COLLAPSE/EXPAND BUTONU — DAİMA GÖRÜNÜR === */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {{
        visibility: visible !important;
        opacity: 1 !important;
        display: flex !important;
        z-index: 999999 !important;
    }}

    /* Sidebar kapalıyken sol üstte açma butonu */
    [data-testid="collapsedControl"] {{
        background: {theme['bg_secondary']} !important;
        border: 1px solid {theme['border_light']} !important;
        border-radius: 10px !important;
        padding: 8px 12px !important;
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
    }}

    [data-testid="collapsedControl"]:hover {{
        background: {theme['accent_primary']} !important;
        border-color: {theme['accent_primary']} !important;
        transform: scale(1.05);
    }}

    [data-testid="collapsedControl"]:hover svg {{
        color: white !important;
        fill: white !important;
    }}

    /* Sidebar açıkken kapatma butonu içeride sağ üstte */
    [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {{
        position: absolute !important;
        top: 0.5rem !important;
        right: 0.5rem !important;
        background: transparent !important;
        border: none !important;
    }}
    </style>
    """

    return css


def load_css() -> None:
    """Inject theme-aware CSS. CRITICAL: çağrı sırası — set_page_config sonrası,
    HER markdown'dan ÖNCE."""
    theme = get_current_theme()
    css = _build_css(theme["name"])

    # Ana tema CSS'i
    st.markdown(css, unsafe_allow_html=True)

    # ÇİFT GÜVENCE: Streamlit'in kendi CSS variable'larını DA Apple renkleriyle override et
    override_streamlit_vars = f"""
    <style>
    /* Streamlit native CSS vars override (çift güvence) */
    :root {{
        --primary-color: {theme['accent_primary']} !important;
        --background-color: {theme['bg_primary']} !important;
        --secondary-background-color: {theme['bg_secondary']} !important;
        --text-color: {theme['text_primary']} !important;
    }}

    /* Body element'i direkt zorla (en agresif override) */
    body, .stApp, [data-testid="stAppViewContainer"] {{
        background-color: {theme['bg_primary']} !important;
        color: {theme['text_primary']} !important;
    }}

    /* Main content area */
    .main .block-container {{
        background-color: {theme['bg_primary']} !important;
        padding-top: 2rem !important;
        max-width: 1400px;
    }}
    </style>
    """
    st.markdown(override_streamlit_vars, unsafe_allow_html=True)


@st.cache_data(ttl=30)
def _cached_db_stats(db_path: str) -> dict:
    """Sidebar quick-stats — 30-second cache so navigation doesn't re-query the DB."""
    from src.fault.database import FaultDatabase

    db = FaultDatabase(db_path)
    return db.stats()


def render_sidebar() -> None:
    """Render the branded sidebar with quick DB stats + theme toggle."""
    init_theme()

    with st.sidebar:
        # BEDAŞ logo en üstte (yoksa atla — başlık zaten kimliği taşır).
        logo_html = logo_img_html(135, container=True, margin="0 auto 0.85rem auto")
        if logo_html:
            st.markdown(logo_html, unsafe_allow_html=True)

        st.markdown("# ⚡ HHATS")
        st.markdown("**Havai Hat Arıza Tespit Sistemi**")
        st.markdown("---")

        # TEMA TOGGLE — Apple-style native st.toggle (Streamlit 1.31+)
        theme = get_current_theme()
        is_dark = st.session_state.theme == "dark"

        col_label, col_toggle = st.columns([2, 1])
        with col_label:
            st.markdown(
                f"<div style='padding-top:6px; font-size:0.95rem; "
                f"font-weight:500; color:{theme['text_primary']};'>"
                f"{'🌙' if is_dark else '☀️'} Tema</div>",
                unsafe_allow_html=True,
            )
        with col_toggle:
            new_value = st.toggle(
                "Koyu",
                value=is_dark,
                key="theme_toggle_switch",
                label_visibility="collapsed",
            )
            if new_value != is_dark:
                toggle_theme()
                st.rerun()

        st.markdown("---")
        st.markdown("*v1.0 - 2026*")
        st.markdown("---")

        try:
            stats = _cached_db_stats(str(DB_PATH))
            critical = stats['by_risk_level'].get('critical', 0)
            high = stats['by_risk_level'].get('high', 0)
            st.markdown("### 📊 Hızlı Durum")
            st.markdown(f"**Toplam Arıza:** {stats['total']}")

            # Kritik — tıklanabilir (Faults'a kritik filtreyle git)
            if st.button(f"🔴 Kritik: {critical}",
                         key="sidebar_critical_btn", width='stretch'):
                st.session_state['faults_default_risk'] = ['critical']
                st.switch_page("pages/3_Faults.py")

            # Yüksek — tıklanabilir
            if st.button(f"🟠 Yüksek: {high}",
                         key="sidebar_high_btn", width='stretch'):
                st.session_state['faults_default_risk'] = ['high']
                st.switch_page("pages/3_Faults.py")
        except Exception:
            st.warning("DB bağlantı hatası")

        # Logged-in user block: identity, change-password, logout, admin link.
        try:
            from src.auth.session import render_account_sidebar
            render_account_sidebar()
        except Exception:
            pass


def render_clickable_card(icon: str, title: str, description: str,
                          nav_key: str, color: str) -> None:
    """Doğrudan tıklanabilir Apple-style kart (buton YOK).

    HTML <a> ile sarılı div — ``?nav=<key>`` query param'ı set eder; main()
    bu param'ı okuyup ilgili sayfaya ``st.switch_page`` yapar.
    """
    theme = get_current_theme()

    st.markdown(f"""
    <a href="?nav={nav_key}" target="_self" class="nav-card-link">
        <div class="nav-card-direct" style="border-left: 4px solid {color};">
            <div style="display:flex; align-items:flex-start; gap:1.25rem;">
                <div style="font-size:2.75rem; line-height:1; flex-shrink:0;">{icon}</div>
                <div style="flex:1;">
                    <h3 style="color:{theme['text_primary']} !important;
                               margin:0 0 0.5rem 0;
                               font-size:1.35rem; font-weight:600;
                               letter-spacing:-0.02em;">
                        {title}
                    </h3>
                    <p style="color:{theme['text_secondary']};
                              margin:0; font-size:0.95rem;
                              line-height:1.5;">
                        {description}
                    </p>
                </div>
                <div style="color:{color}; font-size:1.5rem; align-self:center;">›</div>
            </div>
        </div>
    </a>
    """, unsafe_allow_html=True)


def main() -> None:
    """Landing page — clickable navigation cards."""
    st.set_page_config(
        page_title="HHATS - Havai Hat Arıza Tespit Sistemi",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_theme()
    load_css()

    # GİRİŞ KAPISI — giriş yapılmamışsa login formu gösterilir + st.stop().
    from src.auth.session import require_login
    require_login()

    # QUERY PARAM ile yönlendirme (doğrudan kart tıklamasından gelir)
    nav_target = st.query_params.get("nav", None)
    if nav_target:
        st.query_params.clear()  # tekrar tetiklenmesin diye temizle
        nav_map = {
            "dashboard": "pages/1_Dashboard.py",
            "faults": "pages/3_Faults.py",
            "map": "pages/2_Map.py",
            "reports": "pages/4_Reports.py",
        }
        if nav_target in nav_map:
            try:
                st.switch_page(nav_map[nav_target])
            except Exception:
                # AppTest izole ortamında switch_page çalışmaz, ignore
                pass

    render_sidebar()

    theme = get_current_theme()

    # Başlık
    st.markdown(
        '<h1 class="main-title">⚡ Havai Hat Arıza Tespit Sistemi</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="subtitle" style="color:{theme["text_secondary"]};">'
        f'Drone tabanlı yapay zeka destekli arıza tespit ve raporlama platformu</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Hoşgeldiniz başlığı
    st.markdown(
        f"<h2 style='color:{theme['text_primary']} !important; font-weight:600;'>"
        f"👋 Hoşgeldiniz</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<p style='color:{theme['text_secondary']}; margin-bottom:2rem;'>"
        f"Aşağıdaki kartlardan birine tıklayarak başlayın:</p>",
        unsafe_allow_html=True,
    )

    # 4 KART GRID (2x2) — DOĞRUDAN TIKLANABILIR, BUTONSUZ
    row1_c1, row1_c2 = st.columns(2)
    with row1_c1:
        render_clickable_card(
            icon="📊", title="Dashboard",
            description="Genel sistem durumu, KPI'lar ve istatistikler",
            nav_key="dashboard", color=theme['accent_primary'],
        )
    with row1_c2:
        render_clickable_card(
            icon="🚨", title="Arıza Yönetimi",
            description="Detaylı arıza listesi, filtreleme, görüntü inceleme",
            nav_key="faults", color=theme['danger'],
        )

    st.markdown("<br>", unsafe_allow_html=True)

    row2_c1, row2_c2 = st.columns(2)
    with row2_c1:
        render_clickable_card(
            icon="🗺️", title="Harita Görünümü",
            description="Arızaların coğrafi dağılımı, risk renkli işaretler",
            nav_key="map", color=theme['success'],
        )
    with row2_c2:
        render_clickable_card(
            icon="📄", title="Raporlar",
            description="PDF ve Excel formatında rapor oluşturma",
            nav_key="reports", color=theme['orange'],
        )

    st.markdown("<br><br>", unsafe_allow_html=True)

    # Sistem bilgi notu
    st.markdown(
        f"""
        <div style="background:{theme['bg_tertiary']};
                    border:1px solid {theme['border_light']};
                    border-left:4px solid {theme['accent_primary']};
                    border-radius:12px;
                    padding:1rem 1.5rem;
                    color:{theme['text_primary']};">
            <strong>ℹ️ Sistem Hakkında:</strong> Bu platform, drone üzerinden RGB
            ve termal kamera ile havai hat ekipmanlarını analiz eder, yapay zeka
            ile arıza tespit eder ve risk seviyesine göre raporlar.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
