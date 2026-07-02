---
name: system-vision
description: Nihai sistem vizyonu - 3 katmanlı saha mimarisi (algılama/işleme/raporlama)
metadata:
  type: project
---

**Nihai hedef:** Drone üzerinden canlı RGB + termal görüntü işleme, anlık arıza tespiti, GPS-tabanlı raporlama.

**3 Katmanlı Mimari:**

**Katman 1 - Algılama (Sense):** Drone (DJI Mavic 2 Enterprise Advanced)
- RGB kamera (30 FPS, 5472x3648)
- Termal kamera (30 FPS, 640x512, radyometrik)
- GPS, IMU, yükseklik, yön bilgisi
- Eş zamanlı dual-sensor capture

**Katman 2 - İşleme (Process):** Yer İstasyonu
- Seçenek A: Drone üzerinde (zor, küçük payload)
- Seçenek B (önerilen): Yerde Raspberry Pi + AI accelerator
- Drone video stream → DJI OcuSync → Pi
- Anlık YOLO inference + termal hotspot detection
- GPS eşleştirme, risk skorlama

**Katman 3 - Raporlama (Report):** Saha + Bulut
- Lokal SQLite veritabanı
- Tablet/mobil canlı harita (Folium/Mapbox)
- Otomatik PDF raporu (uçuş sonrası)
- Opsiyonel bulut senkronizasyonu
- Renk kodlu marker: yeşil/sarı/turuncu/kırmızı (risk seviyesi)

**Saha senaryosu:** Operatör tabletinde harita açık, drone uçuyor. Her arıza tespit edildiğinde bildirim: "Polymer izolatörde hotspot - 41.123N 28.456E - kritik risk". Uçuş bitince otomatik PDF.

**Akademik değer:** Bitirme tezi + ileride patent/startup potansiyeli. Türkiye dağıtım şebekesi (BEDAŞ vb.) için pratik değer.

**Bağlı dosyalar:** [[mvp-roadmap]] - 6 MVP'lik geliştirme planı, MVP-6 = Raspberry Pi entegrasyonu.
