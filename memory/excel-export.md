---
name: excel-export
description: Excel (.xlsx) export utility — formatlı 3-sheet rapor üretici (openpyxl)
metadata:
  type: project
---

**`src/dashboard/utils/excel_export.py`** (MVP-5.3, 2026-06-02) — `export_faults_to_excel(df, filename=None) -> bytes`. Filtrelenmiş arıza DataFrame'ini renkli, formatlı 3-sheet Excel'e döker; `bytes` döner → `st.download_button(data=...)` ile kullanılır. openpyxl 3.1.5 (zaten kurulu).

**Sheet'ler:** (1) **Arızalar** — başlık/rapor tarihi/toplam + 8 kolonlu tablo (Fault ID/Tarih/Uçuş/Arıza Tipi/Ekipman/Risk/Skor/Durum); Risk + Durum hücreleri renk dolgulu (RISK_FILL_COLORS / STATUS_FILL_COLORS hex), `freeze_panes='A6'`. (2) **Risk Özeti** — risk seviyesi adet+yüzde, renkli. (3) **Uçuş Özeti** — `df.groupby('flight_label').agg(total/critical/high)` + other.

**Beklenen df kolonları:** `fault_id, fault_id_short, timestamp, flight_label, fault_type_label, equipment_label, risk_level, risk_level_tr, risk_score, status, status_tr`. Faults sayfasının `filtered` df'i hepsini içerir; Dashboard df'i farklı şema → export öncesi türetilir (fault_id_short, flight_label=FLIGHT_SHORT map, equipment_label=equipment_class ham, risk_level_tr=RISK_TR map, status_tr=status_label).

**Kullanım:** `3_Faults.py` filtrelerden sonra "📥 Excel İndir" (filtered) + `1_Dashboard.py` "📥 Tüm Verileri Excel" (tüm df). `st.download_button` her tıklamada rerun YAPMAZ (bilinçli Streamlit davranışı). Helper'lar `_apply_header_style` (koyu bg beyaz yazı) / `_apply_cell_style(bg/font/bold)`.

**DİKKAT — paket çakışması (çözüldü):** Eski `src/dashboard/utils.py` (3-satır kullanılmayan stub) silinip `src/dashboard/utils/` paketine çevrildi (`__init__.py` + `excel_export.py`). Aynı dizinde hem `utils.py` hem `utils/` olamaz. Import: `from src.dashboard.utils.excel_export import export_faults_to_excel`.

Bağlı: [[dashboard]], [[faults]], [[fault-database]]. Doğrulama: 10 kayıt → 7.6KB, 3 sheet.
