---
name: inspection-pipeline
description: MVP-4.3 uçtan uca inspection pipeline (RGB+termal → fault record → risk → DB)
metadata:
  type: project
---

`scripts/run_inspection_pipeline.py` — bir RGB görüntü (+ opsiyonel termal .npy) alır → YOLO tespit (`best.pt`) → hybrid hotspot detection → `FaultRecord`'lar → `apply_risk_to_record` → `FaultDatabase`'e yazar.

**Mantık:** Her RGB tespit bir kayıt olur; tespitin bbox'ı bir hotspot bbox'ı ile çakışırsa `COMBINED`, yoksa `EQUIPMENT_DETECTED`. Hiçbir tespitle çakışmayan hotspot'lar standalone `HOTSPOT` kaydı. `bbox_overlap()` basit kesişim (area>0). RGB+termal birlikteyse termal, RGB frame'ine `cv2.resize` ile ölçeklenir (sadece demo hizalaması; gerçekte sensör-hizalı RGB/termal gerekir — MVP-1).

**Çalıştırma:** `--demo` (5 InsPLAD test img + 3 sentetik kritik termal) veya `--rgb <path> [--thermal <npy>] [--conf] [--flight-id] [--db]`. Model yoksa/görüntü yoksa net hata.

**Demo sonucu (gerçek inference):** 9 kayıt = 5 equipment_detected (medium) + 4 thermal_hotspot (critical). **COMBINED=0** — çünkü InsPLAD görüntüleri ile sentetik termal GERÇEKTE hizalı değil, hotspot tespit konumuyla çakışmadı (uydurma yok). COMBINED kod yolu ayrıca hizalı veriyle doğrulandı: insulator_polymer + 110°C hotspot → COMBINED/CRITICAL/90.1.

**Çıkarım:** Gerçek COMBINED tespiti için sensör-hizalı RGB+termal çifti şart (kullanıcının kendi drone verisi / MVP-1). Bağlı: [[hotspot-detector]], [[fault-database]], [[mvp-roadmap]] MVP-4→MVP-5, [[teias-ir-dataset]].
