# Memory Index

- [YOLO dataset prep](yolo-dataset-prep.md) — how scripts/prepare_yolo_dataset.py merges 3 sources into data/yolo
- [InsPLAD duplicate filenames](insplad-duplicate-filenames.md) — train JSON has 46 dup file_names; key by file_name to avoid leakage
- [venv Python environment](venv-python-environment.md) — .venv (py3.11) has no pip; missing tqdm/pycocotools; force UTF-8 console
- [YOLO training](yolo-training.md) — scripts/train_yolo.py; Ultralytics relative-project & 8GB VRAM gotchas
- [MVP Roadmap](mvp-roadmap.md) — 6 MVP plan (MVP-2 active, MVP-6 Raspberry Pi added)
- [System Vision](system-vision.md) — 3-layer architecture: sense/process/report; final goal is real-time field deployment
- [Future Improvements](future-improvements.md) — model iyileştirme stratejileri (class weights, TTPLA, kendi veri ile fine-tune)
- [TEİAŞ IR dataset](teias-ir-dataset.md) — MVP-3 termal veri: 4000 IR + 4000 VL, 128x128 BMP, TV=hat var / TY=hat yok
- [Synthetic thermal](synthetic-thermal.md) — generate_synthetic_thermal.py; 200 radyometrik .npy (°C) + ground truth, 10 senaryo
- [Hotspot detector](hotspot-detector.md) — src/thermal 3 yöntem (abs/stat/hybrid); hybrid F1=1.0, risk kalibrasyonu bozuk
- [Fault database](fault-database.md) — MVP-4 FaultRecord + SQLite (tablo=fault_records) + risk scoring; +MVP-5.3 personnel/notes/assignment tabloları + migrate_db_v2.py
- [Inspection pipeline](inspection-pipeline.md) — MVP-4.3 uçtan uca RGB+termal → fault record → risk → DB; --demo
- [Demo data](demo-data.md) — generate_demo_data.py; 102 kayıt, 4 flight (Kocaeli/Sakarya/Bursa/İstanbul), dashboard için
- [Dashboard](dashboard.md) — MVP-5 Streamlit HHATS; AÇIK tema (GitHub/Linear 60/40/10); başlık !important fix; app/1/2/3 sayfa TAM (4 placeholder); cache katmanı (CSS+db_stats+loader); set_page_config/pandas3/streamlit/skor-format gotchaları
- [Faults](faults.md) — MVP-5.3 Arıza Yönetimi (3_Faults.py); filtre/Türkçe-arama/görüntü/status + Aksiyon Merkezi (personel atama/notlar/sil); false_positive ve dotted-İ gotchaları
- [Fault files](fault-files.md) — generate_fault_files.py; klasörlü görsel+metadata paketi; OpenCV non-ASCII path + Türkçe karakter (PIL font) gotchaları
- [Map](map.md) — MVP-5.2 Folium interaktif harita (2_Map.py); GPS marker cluster, filtreler, popup; CartoDB Positron default tile; folium+streamlit-folium
- [Excel export](excel-export.md) — src/dashboard/utils/excel_export.py; openpyxl 3-sheet renkli .xlsx; st.download_button; utils paket çakışması notu
