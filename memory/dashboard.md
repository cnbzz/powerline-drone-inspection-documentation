---
name: dashboard
description: MVP-5 Streamlit dashboard (HHATS) yapısı, çalıştırma ve önemli gotchalar
metadata:
  type: project
---

**HHATS** ("Havai Hat Arıza Tespit Sistemi") — Streamlit multipage dashboard, `outputs/db/faults.db`'den okur (uydurma veri yok), tüm arayüz Türkçe.

**Yapı:**
- `src/dashboard/app.py` — giriş/landing + `load_css()`, `render_sidebar()`, `DB_PATH`, `RISK_COLORS` (paylaşılan; sayfalar buradan import eder)
- `src/dashboard/pages/1_Dashboard.py` — TAM: 8 KPI kartı + 4 grafik (risk pie, status bar, flight stacked bar, type pie) + son kritik arızalar tablosu (renkli) + zaman serisi. pandas/plotly ile, gerçek DB.
- `pages/2_Map.py` — TAM (MVP-5.2): Folium interaktif harita, bkz [[map]].
- `pages/3_Faults.py` — TAM (MVP-5.3): filtre/arama/görüntü inceleme/status güncelleme, bkz [[faults]]. `4_Reports.py` — hâlâ placeholder (MVP-5.4'te dolacak)

**Performans / Cache katmanı (MVP-5.3, 2026-06-02):** Sayfa geçişlerinde tekrarlanan iş cache'lendi. (1) `app.py` `_build_css(theme_name)` `@st.cache_data` — CSS f-string'i tema başına 1 kez kurulur; `load_css()` artık `_build_css(theme['name'])` çağırır. (2) `app.py` `_cached_db_stats(db_path)` `@st.cache_data(ttl=30)` — sidebar quick-stats her sayfada DB'ye gitmiyor (~8ms/nav tasarruf); `render_sidebar()` bunu kullanır. (3) Sayfa loader'ları `@st.cache_data(ttl=60)` (1_Dashboard, 2_Map `load_data`; 3_Faults `load_all_faults` ttl=30). Status update sonrası `st.cache_data.clear()` zorunlu. **Folium map cache YAPILMADI:** `create_map()`+`_repr_html_()` ölçüldü = ~50ms (<200ms eşiği) → gereksiz; DB query zaten cache'li. Streamlit 1.57.0 (`on_select='rerun'`, fragment destekli ama eklenmedi). Net: küçük dataset (102 kayıt) → nav başına ~15ms gereksiz compute elendi (cold→cache hit ~0ms).

**Renk kodu (AÇIK tema — GitHub/Linear, 60/40/10, 2026-05-31 rev):** Önceki Navy/koyu tema TAMAMEN değişti, açık temaya geçildi. %60 dominant ana bg `#fafbfc`; %40 secondary `#eef1f4` (açık gri-mavi — **sidebar bg + tablo thead**; eski #ffffff'di, ana bg'ye çok benzediği için sidebar "kayboluyordu") + tertiary `#ffffff` (saf beyaz — KPI kart bg, ayrışsın diye); kenarlık `#d8dde3`/`#c7cdd4` (eski #e1e4e8'den belirgin); %10 accent mavi `#0969da` + kırmızı `#cf222e`. Metin: primary `#1f2328`, secondary `#656d76`, muted `#8c959f`. **Sidebar AÇIK gri-mavi** (#eef1f4, koyu metin). **Risk renkleri GitHub paleti** (RISK_COLORS app.py): low `#1a7f37`, medium `#bf8700`, high `#bc4c00`, critical `#cf222e`. Plotly: `paper_bgcolor='#ffffff'`, `plot_bgcolor='#fafbfc'`, `font/title_font='#1f2328'`. Bu renkler `app.py` `load_css()` `:root` + `1_Dashboard.py` + `2_Map.py`'de literal geçer — hepsini birlikte güncelle.

**Aşama 4 ekleri (2026-06-03):** (1) **Tıklanabilir dashboard:** `1_Dashboard.py` "Son Kritik Arızalar" altına "🔍 Tüm Kritik Arızaları Görüntüle →" butonu → `st.session_state['faults_default_risk']=['critical']` + `st.switch_page("pages/3_Faults.py")` (Faults bunu tek-seferlik default'a uygular, bkz [[faults]]). (2) **Excel export:** Dashboard "📥 Tüm Verileri Excel" + Faults "📥 Excel İndir", utility [[excel-export]] (`src/dashboard/utils/` PAKETİ — eski `utils.py` stub silindi). (3) **Expander CSS fix:** light tema'da `st.expander` koyu kalıyordu → `_build_css`'e `[data-testid="stExpander"]` (+ summary/details/`*`) tema-bg override + `.stText` metin rengi override. (4) **Detay meta HTML:** Faults'ta `st.text()` görünmezlik sorunu için inline-styled markdown'a geçildi.

**Multiselect tag CSS fix (2026-06-02):** Light tema'da multiselect seçili tag'ları BaseWeb default'u ile koyu bg + kırmızı yazı (okunmuyordu). `_build_css()` sonuna (</style> öncesi) override eklendi: `.stMultiSelect [data-baseweb="tag"]` → accent_primary bg + beyaz yazı + `span{color:white}` + `svg{fill:white}`; ayrıca `[data-baseweb="popover"] li` dropdown öğeleri tema renginde; date/text/select input metni text_primary. BaseWeb internal stilleri için `!important` şart.

**Başlık görünmeme bug fix (2026-05-31):** Streamlit default h1-h6 stilleri `.main-title` rengini eziyor, başlıklar silik görünüyordu. Çözüm: `.main-title { color: #1f2328 !important }` (HARD CODED, var() değil) + global `h1,h2,h3,h4,h5,h6 { color:#1f2328 !important }` + `.stMarkdown h1,h2,h3` + sidebar `h1,h2,h3` override. CSS değişken (var(--text-primary)) override'a yenik düşebiliyor → kritik başlıklarda literal hex + !important kullan.

**Çalıştırma:** `uv run streamlit run src/dashboard/app.py --server.port 8501 --server.headless true` → http://localhost:8501 (health: `/_stcore/health`). Test: home HTTP 200, health ok, 102 kayıt yükleniyor.

**ÖNEMLİ gotchalar (çözüldü):**
1. `set_page_config` `app.py`'de modül seviyesinde DEĞİL `main()` içinde — yoksa sayfalar `load_css` import ederken "set_page_config can only be called once" hatası. Sayfalarda `set_page_config` ilk st çağrısı olmalı (import'lardan sonra ama load_css'ten önce).
2. Her dosyada `sys.path` bootstrap (`parents[2]` app, `parents[3]` pages) — yoksa `import src...` çalışmaz (streamlit sys.path[0]=script dizini yapar).
3. **pandas 3.0**: `Styler.applymap` KALDIRILDI → `Styler.map` kullan.
4. **streamlit**: `use_container_width` deprecated (kaldırma 2025-12-31 geçti) → `width='stretch'` kullan.
5. **Skor format**: tabloda `disp['Skor'].round(1)` yetmez (float "83.600000" gösterir) → `disp.style.format({'Skor': '{:.1f}'}).map(...)` kullan. round() yerine Styler.format().

Veri kaynağı: [[demo-data]] (102 kayıt, 4 flight). Bağlı: [[fault-database]], [[mvp-roadmap]] MVP-5, [[fault-files]] (arıza dosyası üretici), [[map]] (Folium harita sayfası).
