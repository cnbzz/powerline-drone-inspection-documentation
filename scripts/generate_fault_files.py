"""Arıza dosyası üretici (MVP-5 ön hazırlık).

DB'deki her fault record için ayrı bir klasör üretir::

    outputs/fault_files/fault_<uuid8>/
    ├── 01_rgb_original.jpg
    ├── 02_rgb_annotated.jpg     (bbox çizilmiş)
    ├── 03_thermal_original.png  (varsa)
    ├── 04_thermal_annotated.png (hotspot işaretli, varsa)
    ├── 05_combined_view.jpg     (yan yana montaj, ikisi de varsa)
    ├── metadata.json
    └── README.txt               (insan-okunabilir özet)

Çalıştırma::

    uv run python scripts/generate_fault_files.py --clean

Gerçek görüntüleri işler; tahmin/uydurma yoktur. RGB yoksa sadece termal,
termal yoksa sadece RGB dosyaları üretilir. Hatalı kayıt atlanır, loglanır.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Bootstrap project root onto sys.path so ``src...`` imports resolve.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# UTF-8 console (Windows cp1252 fix for Türkçe çıktı).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from src.fault.database import FaultDatabase  # noqa: E402
from src.fault.fault_record import FaultRecord  # noqa: E402

# Renkler (BGR for cv2)
RISK_BBOX_COLORS_BGR = {
    "low": (69, 167, 40),       # yeşil
    "medium": (7, 193, 255),    # sarı
    "high": (20, 126, 253),     # turuncu
    "critical": (53, 53, 220),  # kırmızı
}

CLASS_NAMES_TR = {
    "insulator_polymer": "Polimer İzolatör",
    "insulator_glass": "Cam İzolatör",
    "insulator_porcelain": "Porselen İzolatör",
    "insulator_generic": "İzolatör",
    "tower": "Direk",
    "spacer": "Spacer",
    "damper": "Damper",
    "tower_id_plate": "Direk Plakası",
}


def _imread(path: str | Path) -> "np.ndarray | None":
    """Unicode-safe cv2.imread (Windows cv2.imread bozuk non-ASCII path'lerde)."""
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _imwrite(path: Path, img: "np.ndarray") -> bool:
    """Unicode-safe cv2.imwrite via imencode + tofile."""
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def _get_font(size: int = 24):
    """Windows/Linux için Unicode TrueType font yükle (Türkçe karakter desteği)."""
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        # Linux fallback
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for fp in font_candidates:
        if Path(fp).exists():
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


def draw_bbox_on_rgb(image_path: str, record: FaultRecord, output_path: Path) -> bool:
    """RGB görüntü üzerine bbox + Türkçe etiket çiz (PIL ile Unicode destekli)."""
    if not image_path or not Path(image_path).exists():
        return False

    # OpenCV ile oku (Unicode path-safe)
    img_bgr = _imread(image_path)
    if img_bgr is None:
        return False

    # BGR → RGB → PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    if record.bbox:
        bb = record.bbox
        color_bgr = RISK_BBOX_COLORS_BGR.get(record.risk_level.value, (255, 255, 255))
        # BGR → RGB tuple for PIL
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        # BBox çiz (kalın 4px)
        draw.rectangle(
            [(bb.x, bb.y), (bb.x + bb.width, bb.y + bb.height)],
            outline=color_rgb, width=4
        )

        # Etiket metni
        label = CLASS_NAMES_TR.get(record.equipment_class, record.equipment_class or '?')
        if record.detection_confidence:
            label += f" %{record.detection_confidence * 100:.0f}"

        # Font yükle (Unicode destekli)
        font = _get_font(size=28)

        # Metin boyutu hesapla
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_w = bbox_text[2] - bbox_text[0]
        text_h = bbox_text[3] - bbox_text[1]

        # Etiket arka planı (bbox üstünde)
        label_y = max(bb.y - text_h - 12, 0)
        draw.rectangle(
            [(bb.x, label_y), (bb.x + text_w + 12, label_y + text_h + 10)],
            fill=color_rgb
        )
        # Etiket metni (beyaz)
        draw.text((bb.x + 6, label_y + 4), label, font=font, fill=(255, 255, 255))

        # Risk seviyesi rozeti (sağ üst köşe)
        risk_label = f"RİSK: {record.risk_level.value.upper()} ({record.risk_score:.0f})"
        risk_font = _get_font(size=22)
        rbox = draw.textbbox((0, 0), risk_label, font=risk_font)
        rl_w = rbox[2] - rbox[0]
        rl_h = rbox[3] - rbox[1]

        img_w = pil_img.width
        draw.rectangle(
            [(img_w - rl_w - 24, 10), (img_w - 10, rl_h + 30)],
            fill=color_rgb
        )
        draw.text(
            (img_w - rl_w - 18, 16), risk_label,
            font=risk_font, fill=(255, 255, 255)
        )

    # PIL → RGB array → BGR → kaydet
    img_rgb_out = np.array(pil_img)
    img_bgr_out = cv2.cvtColor(img_rgb_out, cv2.COLOR_RGB2BGR)
    return _imwrite(output_path, img_bgr_out)


def visualize_thermal(thermal_path: str, record: FaultRecord, output_path: Path,
                      annotated: bool = False) -> bool:
    """Termal .npy'yi PNG olarak görselleştir, opsiyonel olarak hotspot işaretle."""
    if not thermal_path or not Path(thermal_path).exists():
        return False

    try:
        thermal = np.load(thermal_path)
    except Exception as e:
        logger.warning(f"Termal yüklenemedi: {thermal_path}: {e}")
        return False

    fig, ax = plt.subplots(figsize=(10, 7.5))
    im = ax.imshow(thermal, cmap="jet", vmin=-10, vmax=150)
    cbar = plt.colorbar(im, ax=ax, label="Sıcaklık (°C)")
    cbar.ax.tick_params(labelsize=10)

    if annotated and record.thermal:
        # Hotspot bölgesini bul (basit: max temp etrafı).
        # Sentetik veride bbox bilgisi kaydedilmedi, ama peak'ten tahmin edebiliriz.
        max_idx = np.unravel_index(np.argmax(thermal), thermal.shape)
        cy, cx = max_idx
        radius = int(np.sqrt(record.thermal.area_pixels / np.pi))

        # Daire çiz
        circle = patches.Circle((cx, cy), radius * 1.2, linewidth=3,
                                edgecolor="white", facecolor="none", linestyle="--")
        ax.add_patch(circle)

        # Etiket
        label = f"HOTSPOT: {record.thermal.peak_temp_c:.1f}°C"
        if record.thermal.temperature_delta_c:
            label += f" (Δ{record.thermal.temperature_delta_c:.1f}°C)"
        ax.annotate(label, xy=(cx, cy), xytext=(cx + 30, cy - 30),
                    color="white", fontsize=12, fontweight="bold",
                    bbox=dict(boxstyle="round", facecolor="red", alpha=0.8),
                    arrowprops=dict(arrowstyle="->", color="white", lw=2))

    risk_text = f"Risk: {record.risk_level.value.upper()} | Skor: {record.risk_score:.0f}"
    ax.set_title(f"Termal Görüntü — {risk_text}", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    return True


def create_combined_view(rgb_annotated: Path, thermal_annotated: Path, output_path: Path) -> bool:
    """RGB + termal yan yana montaj."""
    if not rgb_annotated.exists() or not thermal_annotated.exists():
        return False

    rgb = _imread(rgb_annotated)
    thermal = _imread(thermal_annotated)
    if rgb is None or thermal is None:
        return False

    # Aynı yüksekliğe getir
    target_h = max(rgb.shape[0], thermal.shape[0])
    rgb_resized = cv2.resize(rgb, (int(rgb.shape[1] * target_h / rgb.shape[0]), target_h))
    thermal_resized = cv2.resize(thermal, (int(thermal.shape[1] * target_h / thermal.shape[0]), target_h))

    # Yan yana birleştir (10px boşluk)
    spacer = np.ones((target_h, 10, 3), dtype=np.uint8) * 30  # Navy dark
    combined = np.hstack([rgb_resized, spacer, thermal_resized])

    return _imwrite(output_path, combined)


def write_metadata(record: FaultRecord, fault_dir: Path) -> None:
    """Tüm detayları JSON olarak yaz + insan-okunabilir README."""
    data = record.to_dict()

    readme_lines = [
        f"FAULT REPORT - {record.fault_id}",
        "=" * 60,
        f"Tespit Tarihi   : {record.timestamp}",
        f"Uçuş            : {record.inspection_flight_id or '—'}",
        f"Risk Seviyesi   : {record.risk_level.value.upper()}",
        f"Risk Skoru      : {record.risk_score:.1f}/100",
        f"Arıza Tipi      : {record.fault_type.value}",
        f"Durum           : {record.status.value}",
    ]

    if record.equipment_class:
        readme_lines.append(f"Ekipman         : {CLASS_NAMES_TR.get(record.equipment_class, record.equipment_class)}")

    if record.detection_confidence:
        readme_lines.append(f"Model Güveni    : %{record.detection_confidence * 100:.1f}")

    if record.gps:
        readme_lines.append(f"Konum           : {record.gps.latitude:.6f}°N, {record.gps.longitude:.6f}°E")
        if record.gps.altitude_m:
            readme_lines.append(f"Yükseklik       : {record.gps.altitude_m:.1f}m")

    if record.thermal:
        readme_lines.extend([
            "",
            "TERMAL ANALİZ:",
            f"  Peak Sıcaklık : {record.thermal.peak_temp_c:.1f}°C",
            f"  Ort. Sıcaklık : {record.thermal.mean_temp_c:.1f}°C",
            f"  Ambient       : {record.thermal.ambient_temp_c:.1f}°C",
            f"  Δ Fark        : {record.thermal.temperature_delta_c:.1f}°C",
            f"  Alan          : {record.thermal.area_pixels} piksel",
        ])

    if record.bbox:
        readme_lines.extend([
            "",
            f"BBOX            : x={record.bbox.x}, y={record.bbox.y}, "
            f"w={record.bbox.width}, h={record.bbox.height}",
        ])

    if record.notes:
        readme_lines.extend(["", f"NOT: {record.notes}"])

    if record.reviewer:
        readme_lines.append(f"İnceleyen       : {record.reviewer} ({record.reviewed_at})")

    # JSON
    (fault_dir / "metadata.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # README
    (fault_dir / "README.txt").write_text(
        "\n".join(readme_lines),
        encoding="utf-8",
    )


def process_fault(record: FaultRecord, output_root: Path) -> dict:
    """Bir fault için tüm dosyaları üret. İstatistik döner."""
    fault_dir = output_root / f"fault_{record.fault_id[:8]}"
    fault_dir.mkdir(parents=True, exist_ok=True)

    stats = {"rgb": False, "rgb_annotated": False, "thermal": False,
             "thermal_annotated": False, "combined": False}

    # 01: Orijinal RGB (kopyala)
    if record.rgb_image_path and Path(record.rgb_image_path).exists():
        shutil.copy(record.rgb_image_path, fault_dir / "01_rgb_original.jpg")
        stats["rgb"] = True

    # 02: BBox'lı RGB
    if stats["rgb"]:
        if draw_bbox_on_rgb(record.rgb_image_path, record, fault_dir / "02_rgb_annotated.jpg"):
            stats["rgb_annotated"] = True

    # 03: Orijinal termal (PNG visualize)
    if record.thermal_image_path:
        if visualize_thermal(record.thermal_image_path, record,
                             fault_dir / "03_thermal_original.png", annotated=False):
            stats["thermal"] = True

    # 04: Hotspot işaretli termal
    if stats["thermal"]:
        if visualize_thermal(record.thermal_image_path, record,
                             fault_dir / "04_thermal_annotated.png", annotated=True):
            stats["thermal_annotated"] = True

    # 05: Combined view (ikisi de varsa)
    if stats["rgb_annotated"] and stats["thermal_annotated"]:
        if create_combined_view(fault_dir / "02_rgb_annotated.jpg",
                                fault_dir / "04_thermal_annotated.png",
                                fault_dir / "05_combined_view.jpg"):
            stats["combined"] = True

    # Metadata
    write_metadata(record, fault_dir)

    return stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Arıza dosyası üretici")
    parser.add_argument("--clean", action="store_true", help="Mevcut fault_files'ı sil")
    parser.add_argument("--limit", type=int, default=None, help="Maksimum kayıt sayısı (test için)")
    args = parser.parse_args()

    output_root = Path("outputs/fault_files")

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)
        logger.info(f"Cleaned: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    db = FaultDatabase("outputs/db/faults.db")
    records = db.list_all(limit=args.limit or 10000)

    print(f"\nProcessing {len(records)} fault records...")

    total_stats = {"rgb": 0, "rgb_annotated": 0, "thermal": 0,
                   "thermal_annotated": 0, "combined": 0}

    for i, record in enumerate(records, 1):
        try:
            stats = process_fault(record, output_root)
        except Exception as e:
            logger.error(f"Kayıt atlandı {record.fault_id[:8]}: {e}")
            continue
        for k, v in stats.items():
            if v:
                total_stats[k] += 1
        if i % 20 == 0:
            print(f"  [{i}/{len(records)}] processed")

    print(f"\n{'=' * 60}")
    print("FAULT FILES GENERATED")
    print(f"{'=' * 60}")
    print(f"Total folders     : {len(records)}")
    print(f"With RGB original : {total_stats['rgb']}")
    print(f"With RGB bbox     : {total_stats['rgb_annotated']}")
    print(f"With thermal      : {total_stats['thermal']}")
    print(f"With thermal bbox : {total_stats['thermal_annotated']}")
    print(f"With combined     : {total_stats['combined']}")
    print(f"\nOutput: {output_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
