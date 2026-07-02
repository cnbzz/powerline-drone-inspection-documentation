---
name: teias-ir-dataset
description: TEİAŞ Powerline IR+VL dataset, MVP-3 için gerçek termal veri kaynağı
metadata:
  type: project
---

**Kaynak:** Mendeley Data, Yetgin Ö.E. & Gerek Ö.N. (2017, MIT License)
**URL:** https://data.mendeley.com/datasets/n6wrv4ry6v/8
**Lokal yol:** `data/external/teias_ir_vl/powerline_database/{ir,vl}/`
**İndirme:** Mendeley public-api ile otomatik indirildi (289 MB zip, 302836120 byte). Dosya download_url'leri:
`https://data.mendeley.com/public-api/datasets/n6wrv4ry6v/files?version=8&folder_id=root` → her dosyanın `download_url` alanı (S3 redirect).

**İçerik (DOĞRULANMIŞ):**
- `ir/`  : 4000 görüntü (klasör sadeleştirildi, eski adı "Infrared (IR)")
- `vl/` : 4000 görüntü (eski adı "Visible Light (VL)")
- Boyut: **128x128, RGB, ~48 KB**
- Format: **.bmp** (template'te "JPG/PNG" yazıyordu ama GERÇEKTE BMP — convert gerekebilir)
- İsimlendirme: `TV_IR_0001..2000` + `TY_IR_2001..4000` (VL için aynı)
  - **TV = powerline VAR** (pozitif, 2000 adet), **TY = powerline YOK** (negatif, 2000 adet)
  - Görsel teyit: TV_IR'de çapraz enerji hatları net görünüyor

**Türkiye değeri:** TEİAŞ (Türkiye Elektrik İletim) ile işbirliği. Türk havai hatlarından çekilmiş, sahamıza birebir uygun.

**MVP-3 kullanımı:**
- Hotspot etiketi YOK (binary: powerline var/yok; bbox/segmentasyon yok)
- Algoritmik hotspot detection (OpenCV) için ham veri olarak kullanılacak
- Sentetik mock thermal data ile birlikte test edilecek
- NOT: IR görüntüler radyometrik değil, görsel (RGB BMP) — gerçek sıcaklık değeri içermez

**Bağlı:** [[mvp-roadmap]] MVP-3, [[future-improvements]] termal modelin de fine-tune edilebilir
