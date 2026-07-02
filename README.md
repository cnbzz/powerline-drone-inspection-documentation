# Powerline Drone Inspection

Drone tabanlı havai hat (overhead power line) **AI arıza tespit sistemi**.
RGB ve termal görüntüleri birleştirerek izolatör hasarı, hotspot, eksik
parça, korozyon gibi arızaları tespit eder; coğrafi konumla eşleştirir ve
saha ekipleri için raporlar üretir.

## Özellikler (hedef)

- **RGB Object Detection** — YOLOv8/v11 ile direk, izolatör, conductor,
  damper vb. sınıfların tespiti.
- **Termal Analiz** — R-JPEG ayrıştırma ile gerçek sıcaklık değerleri,
  hotspot tespiti, RGB-termal hizalama.
- **EXIF & GPS** — Görüntüden GPS, altitude ve timestamp çıkarımı.
- **Risk Skorlama** — Konfigüre edilebilir ağırlıklarla arıza ciddiyeti
  değerlendirmesi.
- **Raporlama** — PDF (ReportLab) ve Excel (openpyxl) çıktıları,
  Folium tabanlı interaktif harita.
- **Dashboard** — Streamlit + Plotly tabanlı operatör paneli.

## Kurulum

### 1. Gereksinimler

- Windows 10/11 veya Linux
- Python **3.11**
- [uv](https://docs.astral.sh/uv/) (önerilen) ya da pip
- (Opsiyonel) CUDA destekli GPU — YOLO eğitim/inference için

### 2. Repo'yu klonla

```powershell
git clone <repo-url> powerline-drone-inspection
cd powerline-drone-inspection
```

### 3. Virtual environment ve bağımlılıklar (uv ile önerilen)

```powershell
# uv kurulu degilse:
#   irm https://astral.sh/uv/install.ps1 | iex
uv python install 3.11
uv sync --extra dev
```

> pip ile alternatif:
> ```powershell
> py -3.11 -m venv .venv
> .\.venv\Scripts\Activate.ps1
> pip install -r requirements.txt
> ```

### 4. Ortam değişkenleri

```powershell
Copy-Item .env.example .env
# `.env` dosyasini kendi degerlerinle duzenle
```

### 5. Kurulumu doğrula

```powershell
uv run python scripts/check_environment.py
```

## Klasör Yapısı

```
powerline-drone-inspection/
├── config/              # YAML konfigurasyonlari (model, termal, risk)
├── data/
│   ├── raw/             # Ham drone goruntuleri (git'e girmez)
│   ├── external/        # Public dataset'ler (INSPLAD, MPID, ...)
│   ├── processed/       # Onislenmis goruntu/etiket
│   ├── yolo/            # YOLO formatinda train/val/test
│   └── annotations/     # Kendi etiketlerimiz
├── src/
│   ├── ingestion/       # EXIF, termal parser, RGB-termal eslestirme
│   ├── detection/       # YOLO trainer & inference
│   ├── thermal/         # Hotspot tespiti, sicaklik analizi
│   ├── fault/           # Ariza modeli, risk skorlama, DB
│   ├── reporting/       # PDF & Excel jeneratorleri
│   ├── dashboard/       # Streamlit uygulamasi
│   └── utils/           # Logger, geo, gorselleştirme
├── notebooks/           # Kesifsel Jupyter notebooks
├── scripts/             # Calistirilan CLI scriptleri
├── tests/               # pytest unit testler
├── models/              # Pretrained ve egitilmis agirliklar
├── outputs/             # Predictions, raporlar, loglar
└── docs/                # Mimari, dataset ve etiketleme rehberleri
```

## MVP Roadmap

| Faz       | İçerik                                                           |
|-----------|------------------------------------------------------------------|
| **MVP-1** | Görüntü/metadata okuma: EXIF + GPS + termal parser + RGB-termal eşleştirme |
| **MVP-2** | RGB object detection: YOLO eğitim + inference pipeline           |
| **MVP-3** | Termal analiz: hotspot tespiti, sıcaklık eşikleri, RGB-termal hizalama |
| **MVP-4** | Arıza kaydı: veri modeli, SQLite DB, risk skorlama               |
| **MVP-5** | Raporlama: PDF/Excel/harita çıktıları, Streamlit dashboard       |

Detaylı mimari için `docs/architecture.md` dosyasına bakınız.

## Lisans

Proprietary — tüm hakları saklıdır.
