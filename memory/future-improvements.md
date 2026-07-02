---
name: future-improvements
description: Modeli iyileştirmek için ileride uygulanacak stratejiler ve sıralaması
metadata:
  type: project
---

Mevcut yolov8m_8class_v1 modelinin val performansı:
- Overall mAP50=0.798, mAP50-95=0.664 (genel mükemmel)
- Zayıf 3 sınıf: tower (0.49), insulator_generic (0.39), spacer (0.30) - tümü az veriden (250-350 örnek)

İleride iyileştirme stratejileri (öncelik sırasıyla):

**Strateji 1 - Kendi drone verisiyle fine-tuning (EN ÖNEMLİ):**
- Kullanıcı kendi DJI Mavic 2 Enterprise Advanced ile uçuş yaptığında
- Türkiye dağıtım hattı ekipmanlarını ekle (BEDAŞ vb.)
- Mevcut best.pt'den fine-tune et (sıfırdan değil)
- 10-50 epoch yeter, küçük learning rate (1e-4)
- Beklenen: domain adaptation, sahaya hazır

**Strateji 2 - Class weights eklemek:**
- Train script'ine class weighting ekle (Ultralytics native destek var)
- Hesaplama: weight_i = sqrt(N_max / N_i)
  - damper: 1.0, polymer: 1.1, glass: 1.1, porcelain: 1.7
  - plate: 5.4, generic: 5.5, spacer: 5.2, tower: 6.1
- Yeniden eğitim ~4 saat
- Beklenen: zayıf 3 sınıf %50-80 iyileşir

**Strateji 3 - TTPLA dataset entegrasyonu:**
- Mendeley'den TTPLA indir (tower segmentation odaklı, dokümandaki 6. veri seti)
- COCO/segmentation formatından YOLO detection'a çevir
- prepare_yolo_dataset.py'a yeni source olarak ekle
- Tower örnekleri: 253 → 2000+ olur
- Beklenen: tower mAP 0.49 → 0.70+

**Strateji 4 - insulator_generic sınıfını kaldırma:**
- STN-PLAD'ın "insulator" etiketlerini polymer veya glass'a remap et
- 8 sınıf → 7 sınıf
- Pragmatik basitleştirme

**Strateji 5 - Augmentation tuning:**
- Mevcut Ultralytics defaults (mosaic, hsv, flip) zaten iyi
- Sadece zayıf sınıfa odaklı oversampling denenebilir
- Risk: overfitting, dikkatli yapılmalı

**Aktivasyon koşulları:**
- Kullanıcı: "Modeli yetersiz buldum, iyileştirelim" derse → Strateji 1 + 2 birlikte
- Tower performansı kritik olursa → Strateji 3
- Bitirme tezi tamamlanmadan iyileştirme gerekmiyor (mevcut yeterli)
