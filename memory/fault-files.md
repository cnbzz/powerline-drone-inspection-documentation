---
name: fault-files
description: scripts/generate_fault_files.py — her fault için klasörlü görsel+metadata paketi üretir (MVP-5 ön hazırlık)
metadata:
  type: project
---

**`scripts/generate_fault_files.py`** — DB'deki her `FaultRecord` için `outputs/fault_files/fault_<uuid8>/` klasörü üretir. Gerçek görüntüleri işler, uydurma yok. MVP-5 ön hazırlığı (rapor/inceleme paketi).

**Klasör içeriği (üretilen dosyalar koşula bağlı):**
- `01_rgb_original.jpg` — orijinal RGB (shutil.copy)
- `02_rgb_annotated.jpg` — bbox + etiket + risk rozeti çizilmiş (PIL ile, risk rengine göre)
- `03_thermal_original.png` — termal `.npy` → matplotlib jet colormap (vmin=-10, vmax=150)
- `04_thermal_annotated.png` — peak konumu daire ile işaretli + Δ etiketi
- `05_combined_view.jpg` — RGB+termal yan yana (yalnız ikisi de varsa)
- `metadata.json` — `record.to_dict()` tam dökümü
- `README.txt` — Türkçe insan-okunabilir özet

**Çalıştırma:** `uv run python scripts/generate_fault_files.py --clean` (`--limit N` test için). `--clean` mevcut çıktıyı siler.

**Veri gerçeği (102 kayıt):** kayıtlar ya RGB+bbox (82) ya da termal+thermal_data (20); HİÇBİRİNDE ikisi birden yok → `combined` her zaman 0. RGB yoksa sadece termal, termal yoksa sadece RGB üretilir. Hatalı kayıt atlanır+loglanır, döngü durmaz.

**KRİTİK gotcha — OpenCV + non-ASCII path:** Proje yolu "Ömer Kürşat Canbaz" içeriyor. Windows'ta `cv2.imread`/`cv2.imwrite` Unicode path'lerde SESSİZCE başarısız (None döner / yazmaz), `shutil.copy` ve numpy/matplotlib sorunsuz. Çözüm: scriptteki `_imread` (np.fromfile + cv2.imdecode) ve `_imwrite` (cv2.imencode + tofile) helper'larını kullan — düz `cv2.imread/imwrite` ASLA kullanma. Aynı tuzak görsel işleyen tüm scriptlerde geçerli.

**KRİTİK gotcha — Türkçe karakter bbox etiketi:** `cv2.putText` SADECE ASCII destekler → "Cam İzolatör" → "Cam ??zolat??r". Çözüm: `draw_bbox_on_rgb` PIL tabanlı yeniden yazıldı: `_imread`(BGR) → cv2.cvtColor BGR→RGB → `Image.fromarray` → `ImageDraw` ile kutu+metin (Unicode!) → np.array → RGB→BGR → `_imwrite`. Font: `_get_font()` helper Windows `C:/Windows/Fonts/arial.ttf` (Linux'ta DejaVu/Liberation) TrueType yükler, bulamazsa `ImageFont.load_default()` (bitmap, İ bozuk). `from PIL import Image, ImageDraw, ImageFont` (Pillow streamlit ile zaten gelir). Doğrulandı: "Cam İzolatör %97" + "RİSK: MEDIUM" net render.

**Diğer:** UTF-8 console (`sys.stdout.reconfigure`), matplotlib `Agg` backend (headless), `sys.path` bootstrap (`parents[1]`). RISK_BBOX_COLORS_BGR risk seviyesine göre kutu rengi.

Veri kaynağı: [[demo-data]], model: [[fault-database]]. Görsel arayüz: [[dashboard]]. Bağlı: [[mvp-roadmap]] MVP-5.
