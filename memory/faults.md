---
name: faults
description: MVP-5.3 Arıza Yönetimi sayfası (3_Faults.py) — filtre/arama/görüntü/status
metadata:
  type: project
---

**`src/dashboard/pages/3_Faults.py`** — MVP-5.3'te placeholder'dan TAM uygulamaya geçti. Filtrelenebilir arıza listesi + RGB/termal görüntü inceleme + DB status güncelleme. Veri: `outputs/db/faults.db` (`load_all_faults()` @st.cache_data ttl=30). Bağlı: [[dashboard]], [[fault-database]], [[fault-files]], [[demo-data]].

**Yapı:**
- **Filtreler** (st.expander, açık): serbest metin arama, Risk (multiselect), Durum (multiselect), Uçuş (selectbox), Arıza Tipi (multiselect), Sıralama (Risk Skoru ↓↑ / Tarih Yeni-Eski).
- **4 metrik kartı:** Görüntülenen / Kritik / Çözülmemiş / Bu Ay (render_metric_card, 2_Map'tekiyle aynı stil).
- **Excel İndir** (filtrelerden sonra, tablodan önce): "📥 Excel İndir" download_button → `export_faults_to_excel(filtered)`, bkz [[excel-export]].
- **Tablo:** Styler ile renkli Risk + Durum rozetleri (`.style.format({'Skor':'{:.1f}'}).map(...)`), `st.dataframe(on_select='rerun', selection_mode='single-row')` ile satır seçimi.
- **Pagination:** >50 kayıtta `st.number_input` ile sayfalama (PAGE_SIZE=50). Seçim `page_df.iloc[selected_idx]` (gösterilen sayfaya göre pozisyonel) — bu yüzden `filtered.reset_index(drop=True)` ŞART.
- **Risk filtresi dinamik default (Aşama 4):** Dashboard'dan "Tüm Kritik Arızaları Görüntüle" ile gelince `st.session_state['faults_default_risk']` okunur, multiselect default'una uygulanır, sonra silinir (tek seferlik). risk_filter multiselect'te KEY YOK (default'un her switch_page'de uygulanması için).
- **Detay paneli** (satır seçilince): risk badge + 3 sütun görüntü (`01/02_rgb`, `04_thermal` — `outputs/fault_files/fault_<short_id>/`'dan; **her görüntü altında "📥 İndir" download_button**, zoom hint caption; yoksa st.info) + 2 sütun metadata (**HTML force-styled** — bkz CSS gotcha) + **Aksiyon Merkezi** (aşağıda).

**Aksiyon Merkezi (MVP-5.3, 2026-06-02)** — `render_detail_panel()` içindeki eski 4-buton bölümü tamamen yenilendi, 4 alt-bölüm:
1. **🎯 Durum Yönetimi:** mevcut durum rozeti + 3 buton (İncelendi/Çözüldü/Yanlış Alarm), `update_fault_status()`.
2. **👤 Personel Ataması:** mevcut atama kartı + personel selectbox (None=atamayı kaldır) + öncelik (Düşük/Normal/ACİL) + 💾 Atama Kaydet (`db.assign_fault`) + "➕ Yeni Personel Ekle" expander (`db.add_personnel`). Cache: `load_personnel_list()` ttl=60, `load_fault_assignment()` ttl=10.
3. **📝 İnceleme Notları:** mevcut notlar (renkli kart, tip rozeti 🔧Teknik/🌍Saha/👔Yönetim) + her notta 🗑️ sil + yeni not formu (text_area + tip + yazan + 📝 Not Ekle). Cache: `load_fault_notes()` ttl=10. `db.add_note/list_notes/delete_note`.
4. **🗂️ Diğer İşlemler (Aşama 4'te değişti):** sol=görüntü yol bilgisi (caption); sağ=**⏸️ Pasife Al** (soft delete, 2 adımlı onay → `db.deactivate_fault`). ESKİ "📁 Klasörü Aç" (file:/// linki tarayıcı güvenliği nedeniyle çalışmıyordu) ve "🗑️ Sil" (hard delete) KALDIRILDI. Pasife alınan kayıt `list_all` default'undan gizlenir, veri korunur.

Her DB değişikliğinden sonra `st.cache_data.clear()` + `st.rerun()`. DB metotları [[fault-database]]'de (soft delete: `deactivate_fault`/`reactivate_fault`). Yazan default = "Ömer Kürşat Canbaz".

**KRİTİK gotcha — st.text() metni görünmüyor (Aşama 4 fix):** Light tema'da detay panelindeki `st.text()` metinleri (Temel Bilgiler / Konum & Termal altındaki değerler) kontrastsız/görünmez kaldı. Çözüm: `st.text()` yerine inline-styled `st.markdown(... <div style="color:{theme[text_primary]}; ...">`. Ek olarak `app.py` `_build_css`'e `.stText/[data-testid="stText"]` + expander (`[data-testid="stExpander"]`) override eklendi (light tema'da expander koyu kalıyordu). bkz [[dashboard]].

**KRİTİK gotcha — FaultStatus value'su `false_positive` (false_alarm DEĞİL):** Enum `FaultStatus.FALSE_POSITIVE = "false_positive"`. Görev şablonu yanlışlıkla `false_alarm` kullanıyordu; status filtre options, buton ve `update_fault_status` hepsi `false_positive` ile çalışmalı yoksa `FaultStatus("false_alarm")` ValueError verir. STATUS_TR: `false_positive`→"Yanlış Alarm".

**KRİTİK gotcha — Türkçe dotted-İ arama:** "Cam **İ**zolatör"deki İ (U+0130) ASCII 'i'ye case-fold OLMAZ → `str.contains("izolat", case=False)` 0 sonuç döner. Çözüm: `_tr_normalize()` (str.translate ile İ/I/ı/Ş/Ğ/Ü/Ö/Ç→ascii + lower) + `load_all_faults()` içinde önceden hesaplanan `search_blob` kolonu; arama `search_blob.str.contains(_tr_normalize(q), regex=False)`. Doğrulandı: "izolat"→57 izolatör.

**Status update DB'ye yazıyor:** `FaultDatabase.update_status(fault_id, FaultStatus(x), reviewer="dashboard_user")` → SQLAlchemy SQLEnum value'yu enum ADI olarak saklar (DB'de status="REVIEWED"), okurken `.value` ile "reviewed" döner. Update sonrası `st.cache_data.clear()` ŞART (load_all_faults + sidebar stats tazelensin). SQL ile doğrulandı.

**Demo veri notları:** "izolat"→57 (glass 25/polymer 22/porcelain 10), "termal"→20 hotspot, critical+new = **9** (şablonun "~16" tahmini yanlış, gerçek 9). 20 fault klasöründe termal görüntü var (04_), 102 klasörde RGB.

**Not (uygulanmadı):** Görev şablonunda Tarih aralığı (date_input) ve Sil (delete) bahsedildi ama KOD ŞABLONU'na dahil değildi → eklenmedi. İleride istenirse `FaultDatabase.delete()` mevcut.
