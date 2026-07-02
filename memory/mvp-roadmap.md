---
name: mvp-roadmap
description: Projenin MVP (Minimum Viable Product) yol haritası - MVP-1 ile MVP-6 arası
metadata:
  type: project
---

Drone tabanlı havai hat AI arıza tespit sistemi 6 MVP halinde planlandı:

**MVP-1: Ingestion** (ertelendi - kendi drone verisi yok henüz)
- EXIF reader, thermal R-JPEG parser
- RGB-thermal pairing, video frame extractor
- Kullanıcı kendi drone verisi topladığında uygulanacak

**MVP-2: RGB Object Detection** (aktif - eğitim çalışıyor)
- 8-sınıflı YOLOv8m detector (insulator_polymer/glass/porcelain/generic, tower, spacer, damper, tower_id_plate)
- 3 dataset birleşik: InsPLAD-det (10561), STN-PLAD (133), MPID (5019)
- Train 11262, val 3141, test 1310
- 50 epoch, batch 16, imgsz 640, RTX 4060 GPU

**MVP-3: Termal Hotspot Detection** (sırada)
- OpenCV tabanlı (deep learning gerektirmiyor ilk versiyonda)
- DJI Thermal SDK ile radyometrik veri okuma
- Eşik tabanlı hotspot tespiti, RGB ile alignment

**MVP-4: Fault Database + Risk Scoring**
- SQLite veritabanı, FaultRecord dataclass
- Risk skoru formülü (visual + thermal + context)
- GPS koordinatı ile eşleştirme

**MVP-5: Reporting + Dashboard**
- Streamlit + Folium harita
- PDF raporu (ReportLab)
- Excel arıza listesi (openpyxl)

**MVP-6: Edge Deployment - Raspberry Pi** (YENİ EKLENDİ)
- Kullanıcının vizyonu: sahada anlık inference
- Donanım: Raspberry Pi 5 (8GB) + Hailo-8L AI Kit (önerilen) veya Coral USB
- Model optimizasyon: ONNX export + INT8 quantization
- Drone bağlantısı: DJI OcuSync video stream
- Yer istasyonu mimarisi (Seçenek B): drone → Pi (yerde) → tablet
- Tahmini maliyet: ~10.500 TL
- Bitirme tezi için OPSİYONEL, "gelecek çalışma" olarak yazılabilir

Öncelik sırası: MVP-2 (mevcut) → MVP-3 → MVP-4 → MVP-5 → MVP-1 (kendi veri) → MVP-6
