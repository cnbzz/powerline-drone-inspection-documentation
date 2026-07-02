---
name: hotspot-detector
description: MVP-3 OpenCV hotspot detector (3 yöntem) ve sentetik veri değerlendirme bulguları
metadata:
  type: project
---

`src/thermal/hotspot_detector.py` — radyometrik termal (°C, 2-D float32) üzerinde 3 yöntem + birleşik API `detect_hotspots(thermal, method=...)`. Her yöntem connectedComponents → `Hotspot(bbox, peak/mean_temp_c, area_pixels, risk_level, confidence)` döner; `DetectionResult` içinde toplanır. `bbox_iou()` helper'ı değerlendirme için.

- **absolute**: sabit eşik (low=60, med=80, high=100°C)
- **statistical**: median + k·std outlier (sigma=3, min_temp_diff=15)
- **hybrid** (varsayılan): HEM absolute_min=50 HEM statistical (sigma=2.5) — ikisini birden geçmeli

**Değerlendirme:** `scripts/evaluate_hotspot_detector.py --clean` → 200 sentetik görüntü, ~8 sn. Çıktı: `outputs/hotspot_evaluation/` (comparison_table.csv, per_scenario.csv, confusion PNG'ler, false_positives/negatives.json, summary_report.txt, visual_check/).

**Gerçek bulgular (image-level F1):**
- absolute: P=1.00 R=0.75 F1=0.857 — **FN=25**: tüm 10_LoadHeating (55°C faz <60 eşik) + 5 mild kaçtı. Zayıflık: "ılık" arızaları (<60°C) göremez.
- statistical: P=0.86 R=1.00 F1=0.926 — **FP=16**: sadece tuzaklar değil, NORMAL ekipman da (01 PowerLine×4, 02 Tower×5, 07 noise×4, 08 sun×3). Zayıflık: düz sahnede normal sıcak izolatör bile outlier sayılıyor → "ortalamadan sıcak" ≠ "arıza".
- **hybrid: P=1.00 R=1.00 F1=1.00** — absolute floor(50°C) normal ekipman/güneş/noise FP'lerini eler, düşük floor mild + 55°C fazı yakalar. En iyi.

**ÖNEMLİ uyarılar:**
1. F1=1.0 SENTETİK veride; gerçek DJI verisinde ayrılabilirlik daha zor olacak — bu sonuç tavan değil baseline.
2. **Risk-level kalibrasyonu BOZUK** (tüm yöntemlerde): detector'ın risk eşikleri GT'nin senaryo-risk tanımlarıyla uyuşmuyor (GT medium→pred high/critical vb.). Detection mükemmel ama risk etiketleme ayarlanmalı.

Bağlı: [[synthetic-thermal]] (yer-gerçeği kaynağı), [[teias-ir-dataset]], [[mvp-roadmap]] MVP-3.
