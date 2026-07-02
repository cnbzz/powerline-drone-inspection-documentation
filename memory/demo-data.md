---
name: demo-data
description: Dashboard için üretilen çoklu-flight demo fault verisi ve nasıl yeniden üretileceği
metadata:
  type: project
---

`scripts/generate_demo_data.py` → `outputs/db/faults.db`'yi sıfırdan üretir (seed=42, tekrarlanabilir). Gerçek YOLO inference (InsPLAD test görüntüleri) + gerçek hybrid hotspot detection (sentetik termal). Uydurma yok.

**4 flight (Türkiye hat koridorları):**
- `flight_2026_05_01_kocaeli` — Kocaeli-Gebze (40.80, 29.40)
- `flight_2026_05_08_sakarya` — Sakarya-Adapazarı (40.77, 30.40)
- `flight_2026_05_15_bursa` — Bursa-İnegöl (40.18, 29.06)
- `flight_2026_05_22_istanbul` — İstanbul-Çatalca (41.14, 28.46)

Her flight: 10 RGB (conf 0.30) + 4 sentetik termal. GPS bir hat boyunca dağıtılır (`scatter_gps`), tarih flight gününe yayılır.

**Üretilen veri (mevcut):** 102 kayıt (1–22 Mayıs 2026 arası). Hepsinde GPS var.
- Risk: low 7 / medium 61 / high 18 / critical 16 (hepsi temsil)
- Status: new 68 / reviewed 18 / resolved 12 / false_positive 4 (lifecycle çeşitliliği)
- Type: equipment_detected 68 / thermal_hotspot 20 / visual_defect 14
- Flight başına: Kocaeli 22, Sakarya 38, Bursa 24, İstanbul 18

**Not:** Hedef ~40-70 idi ama her RGB görüntüsü ~2 tespit verdiği için 102 oldu (daha zengin, dashboard için iyi). Azaltmak için `IMAGES_PER_FLIGHT` düşürülebilir. medium-ağırlıklı dağılım [[fault-database]] dinamik normalizasyon yan etkisinden (equipment+confidence → medium).

Yeniden üret: `uv run python scripts/generate_demo_data.py`. Bağlı: [[inspection-pipeline]], [[mvp-roadmap]] MVP-5 (dashboard bu DB'yi okuyacak).
