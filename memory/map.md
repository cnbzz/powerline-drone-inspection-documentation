---
name: map
description: MVP-5.2 Folium interaktif harita sayfası (2_Map.py) — GPS arıza haritası, filtreler, popup
metadata:
  type: project
---

**`src/dashboard/pages/2_Map.py`** (MVP-5.2) — Folium tabanlı interaktif arıza haritası. Placeholder'dan tam çalışan sayfaya geçti.

**Paketler:** `folium` (0.20.0, zaten yüklüydü) + `streamlit-folium` (0.27.2, `uv add` ile eklendi). Render: `from streamlit_folium import st_folium`.

**Özellikler:**
- `load_data()` — DB'den GPS'i olan kayıtları çeker (`r.gps is None` olanları atlar). 102 kaydın GPS'li olanları haritada.
- 3 filtre: Risk seviyesi (multiselect), Uçuş (selectbox, "Hepsi" default), Arıza tipi (multiselect) — hepsi `format_func` ile Türkçe etiket.
- `MarkerCluster` (folium.plugins) — yakın noktalar gruplanır.
- Marker renkleri folium-uyumlu isimlerle (DİKKAT: hex değil!): low=green, medium=orange, high=red, critical=darkred. `folium.Icon(color=..., icon=..., prefix='fa')`.
- `build_popup_html()` — HTML popup: risk başlık (renkli), arıza/ekipman/konum/tarih/ID + termal bloğu (peak temp varsa). `folium.Popup(html, max_width=320)`.
- **Tile (2026-05-31):** default `CartoDB positron` (minimal açık SaaS stili, Notion/Linear gibi — eski OpenStreetMap default yoğun renkli olduğu için değişti). `folium.Map(tiles=None)` + manuel `TileLayer` ekleme sırası: Positron ("Sade (Önerilen)") → OpenStreetMap ("Standart") → dark_matter ("Koyu"). `LayerControl(position='topright', collapsed=False)`. Boş-harita fallback'i de Positron.
- Üst: 4 özet metrik kartı (`render_metric_card`, inline HTML). Alt: görüntülenenlerin Styler'lı tablosu (risk renk + skor/enlem/boylam format).

**Renkler:** açık tema GitHub paleti, bkz [[dashboard]] (accent #0969da, risk #1a7f37/#bf8700/#bc4c00/#cf222e). Popup/metrik kartları inline literal hex kullanır.

**Gotcha:** `st_folium(m, width=None, height=600, returned_objects=[])` — `returned_objects=[]` gereksiz rerun'ı engeller (etkileşim state'i geri döndürmeye gerek yok). AppTest (`streamlit.testing.v1`) ile 0 exception render doğrulandı.

Bağlı: [[dashboard]] (ana yapı + tema), [[fault-database]] (GPS alanı), [[demo-data]], [[mvp-roadmap]] MVP-5.2.
