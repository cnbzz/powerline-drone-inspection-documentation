---
name: synthetic-thermal
description: MVP-3 sentetik radyometrik termal veri üreteci ve çıktı formatı
metadata:
  type: project
---

`scripts/generate_synthetic_thermal.py` → `data/synthetic/thermal/` altına 10 senaryo × 20 = **200 örnek** üretir. `--clean` ile sıfırlanır, seed=42 (deterministik).

**Çıktı (her örnek için):**
- `.npy` : **float32, (480,640), gerçek °C değerleri** — ALGORİTMA BUNU OKUR (DJI R-JPEG radyometrik verisinin soyutlaması)
- `.png` : jet colormap + colorbar + bbox overlay — sadece insan için
- `.json`: ground truth (has_hotspot, hotspots[bbox/peak/mean/area/risk], equipment_regions, notes)
- `ground_truth_all.json`: tüm metadata özeti

**10 senaryo:** 01_Normal_PowerLine, 02_Normal_Tower, 03_Mild(45-55°C), 04_Medium(65-80°C), 05_Critical(100-130°C), 06_MultiHotspot, 07_AmbientNoise(σ=5, no fault), 08_SunWarming(false-positive tuzağı, gradient ama hotspot yok), 09_WindCooling(5-10°C ambient), 10_LoadHeating(faz dengesizliği: 35/40/55°C, faz3 flag).

**Denge:** 100 hotspot'lu (sc 03,04,05,06,10) + 100 hotspot'suz (sc 01,02,07,08,09). Sıcaklık aralığı ~-7°C…161°C.

**Amaç:** OpenCV tabanlı hotspot detection algoritmasını bu yer-gerçeğine karşı geliştirip test etmek. 07/08 senaryoları algoritmanın yanlış-pozitif vermemesini sınar. Gerçek DJI radyometrik verisi geldiğinde aynı `.npy float32 °C` arayüzü kullanılacak. Bağlı: [[teias-ir-dataset]], [[mvp-roadmap]] MVP-3.
