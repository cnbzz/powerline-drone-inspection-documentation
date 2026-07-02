"""Populate the fault DB with rich demo data for the MVP-5 dashboard.

Simulates several inspection flights over different Turkish line corridors, each on a
different date, using REAL YOLO inference on InsPLAD test images plus REAL hybrid
hotspot detection on synthetic thermal maps. No fabricated detections.

Usage::

    uv run python scripts/generate_demo_data.py
"""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

from src.fault.database import FaultDatabase  # noqa: E402
from src.fault.fault_record import (  # noqa: E402
    BoundingBox,
    FaultRecord,
    FaultStatus,
    FaultType,
    GpsLocation,
    ThermalData,
)
from src.fault.risk_scorer import apply_risk_to_record  # noqa: E402
from src.thermal.hotspot_detector import detect_hotspots  # noqa: E402

SEED = 42

CLASS_NAMES = {
    0: "insulator_polymer", 1: "insulator_glass", 2: "insulator_porcelain",
    3: "insulator_generic", 4: "tower", 5: "spacer", 6: "damper", 7: "tower_id_plate",
}

# Realistic Turkish line corridors (one per flight).
FLIGHT_REGIONS: list[dict] = [
    {"flight_id": "flight_2026_05_01_kocaeli", "name": "Kocaeli-Gebze Hattı",
     "center_lat": 40.80, "center_lon": 29.40, "date": datetime(2026, 5, 1, 9, 30)},
    {"flight_id": "flight_2026_05_08_sakarya", "name": "Sakarya-Adapazarı Hattı",
     "center_lat": 40.77, "center_lon": 30.40, "date": datetime(2026, 5, 8, 14, 15)},
    {"flight_id": "flight_2026_05_15_bursa", "name": "Bursa-İnegöl Hattı",
     "center_lat": 40.18, "center_lon": 29.06, "date": datetime(2026, 5, 15, 11, 0)},
    {"flight_id": "flight_2026_05_22_istanbul", "name": "İstanbul-Çatalca Hattı",
     "center_lat": 41.14, "center_lon": 28.46, "date": datetime(2026, 5, 22, 10, 45)},
]

IMAGES_PER_FLIGHT = 10   # RGB images per flight
THERMAL_PER_FLIGHT = 4   # synthetic thermal maps per flight


def scatter_gps(center_lat: float, center_lon: float, index: int, total: int) -> GpsLocation:
    """Spread GPS points along a line corridor (a slowly advancing line + jitter)."""
    offset = (index / total) * 0.05  # ~5 km spread
    return GpsLocation(
        latitude=center_lat + offset + random.uniform(-0.002, 0.002),
        longitude=center_lon + offset * 0.5 + random.uniform(-0.002, 0.002),
        altitude_m=random.uniform(75, 110),
    )


def process_flight_rgb(flight: dict, rgb_pool: list[Path], rgb_idx: int,
                       model, db: FaultDatabase) -> tuple[int, int]:
    """Run YOLO on this flight's RGB images, persist records. Returns (new_idx, count)."""
    count = 0
    for i in range(IMAGES_PER_FLIGHT):
        if rgb_idx >= len(rgb_pool):
            break
        rgb_path = rgb_pool[rgb_idx]
        rgb_idx += 1
        gps = scatter_gps(flight["center_lat"], flight["center_lon"], i, IMAGES_PER_FLIGHT)
        timestamp = flight["date"] + timedelta(minutes=i * 3)

        for r in model.predict(str(rgb_path), conf=0.30, verbose=False):
            if r.boxes is None:
                continue
            img_w, img_h = int(r.orig_shape[1]), int(r.orig_shape[0])
            for box in r.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # ~15% of detections flagged as visual_defect for demo variety.
                is_defect = random.random() < 0.15
                fault_type = FaultType.VISUAL_DEFECT if is_defect else FaultType.EQUIPMENT_DETECTED
                record = FaultRecord(
                    timestamp=timestamp,
                    gps=gps,
                    rgb_image_path=str(rgb_path),
                    fault_type=fault_type,
                    equipment_class=CLASS_NAMES.get(cls_id, str(cls_id)),
                    equipment_class_id=cls_id,
                    bbox=BoundingBox(int(x1), int(y1), int(x2 - x1), int(y2 - y1), img_w, img_h),
                    detection_confidence=conf,
                    inspection_flight_id=flight["flight_id"],
                    notes=f"{flight['name']} - otomatik tespit"
                          + (" (görsel hasar şüphesi)" if is_defect else ""),
                )
                apply_risk_to_record(record)
                db.create(record)
                count += 1
    return rgb_idx, count


def process_flight_thermal(flight: dict, thermal_pool: list[Path], thermal_idx: int,
                           db: FaultDatabase) -> tuple[int, int]:
    """Run hybrid hotspot detection on this flight's thermal maps. Returns (new_idx, count)."""
    count = 0
    for j in range(THERMAL_PER_FLIGHT):
        if thermal_idx >= len(thermal_pool):
            break
        thermal_path = thermal_pool[thermal_idx]
        thermal_idx += 1
        gps = scatter_gps(flight["center_lat"], flight["center_lon"],
                          IMAGES_PER_FLIGHT + j, IMAGES_PER_FLIGHT + THERMAL_PER_FLIGHT)
        timestamp = flight["date"] + timedelta(minutes=30 + j * 3)

        thermal = np.load(thermal_path).astype(np.float32)
        result = detect_hotspots(thermal, method="hybrid")
        ambient = float(result.image_stats.get("median", 20.0))
        for hs in result.hotspots:
            record = FaultRecord(
                timestamp=timestamp,
                gps=gps,
                thermal_image_path=str(thermal_path),
                fault_type=FaultType.HOTSPOT,
                thermal=ThermalData(
                    peak_temp_c=hs.peak_temp_c,
                    mean_temp_c=hs.mean_temp_c,
                    ambient_temp_c=round(ambient, 2),
                    area_pixels=hs.area_pixels,
                    temperature_delta_c=round(hs.peak_temp_c - ambient, 2),
                ),
                inspection_flight_id=flight["flight_id"],
                notes=f"{flight['name']} - termal anomali",
            )
            apply_risk_to_record(record)
            db.create(record)
            count += 1
    return thermal_idx, count


def main() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")
    random.seed(SEED)
    np.random.seed(SEED)

    db_path = PROJECT_ROOT / "outputs" / "db" / "faults.db"
    db_path.unlink(missing_ok=True)
    db = FaultDatabase(db_path)

    model_path = PROJECT_ROOT / "models" / "trained" / "yolov8m_8class_v1" / "weights" / "best.pt"
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        sys.exit(1)
    from ultralytics import YOLO
    logger.info("Loading YOLO model...")
    model = YOLO(str(model_path))

    rgb_pool = sorted((PROJECT_ROOT / "data" / "yolo" / "test" / "images").glob("*.jpg"))
    thermal_dir = PROJECT_ROOT / "data" / "synthetic" / "thermal"
    thermal_pool = (
        sorted((thermal_dir / "05_CriticalHotspot_Joint").glob("*.npy"))
        + sorted((thermal_dir / "04_MediumHotspot_Insulator").glob("*.npy"))
        + sorted((thermal_dir / "03_MildHotspot_Connection").glob("*.npy"))
        + sorted((thermal_dir / "06_MultiHotspot").glob("*.npy"))
    )
    if not rgb_pool:
        logger.error("No RGB test images found in data/yolo/test/images")
        sys.exit(1)
    if not thermal_pool:
        logger.error("No synthetic thermal maps found in data/synthetic/thermal")
        sys.exit(1)

    random.shuffle(rgb_pool)
    random.shuffle(thermal_pool)

    print("=" * 60)
    print("DEMO DATA GENERATION")
    print("=" * 60)

    rgb_idx = thermal_idx = 0
    for flight in FLIGHT_REGIONS:
        logger.info(f"Processing flight: {flight['name']}")
        rgb_idx, n_rgb = process_flight_rgb(flight, rgb_pool, rgb_idx, model, db)
        thermal_idx, n_th = process_flight_thermal(flight, thermal_pool, thermal_idx, db)
        logger.info(f"  {flight['flight_id']}: {n_rgb + n_th} records (RGB {n_rgb}, thermal {n_th})")

    # Realistic lifecycle: move some records out of NEW.
    for record in db.list_all(limit=10000):
        roll = random.random()
        if roll < 0.20:
            db.update_status(record.fault_id, FaultStatus.REVIEWED, reviewer="saha_ekibi")
        elif roll < 0.30:
            db.update_status(record.fault_id, FaultStatus.RESOLVED,
                             reviewer="bakim_ekibi", notes="Onarıldı")
        elif roll < 0.35:
            db.update_status(record.fault_id, FaultStatus.FALSE_POSITIVE,
                             reviewer="saha_ekibi", notes="Yanlış alarm")

    # ---- summary ----
    stats = db.stats()
    all_records = db.list_all(limit=10000)
    print("\n" + "=" * 60)
    print("DEMO DATA SUMMARY")
    print("=" * 60)
    print(f"Total records: {stats['total']}")
    print(f"\nFlights: {len(FLIGHT_REGIONS)}")
    for flight in FLIGHT_REGIONS:
        count = len(db.list_all(flight_id=flight["flight_id"], limit=10000))
        print(f"  {flight['name']:30}: {count} records")
    print("\nBy risk level:")
    for level in ["low", "medium", "high", "critical"]:
        print(f"  {level:10}: {stats['by_risk_level'].get(level, 0)}")
    print("\nBy status:")
    for status, count in stats["by_status"].items():
        print(f"  {status:15}: {count}")
    type_counts: dict[str, int] = {}
    for r in all_records:
        type_counts[r.fault_type.value] = type_counts.get(r.fault_type.value, 0) + 1
    print("\nBy fault type:")
    for ft, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {ft:20}: {count}")
    print(f"\nDB: {db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
