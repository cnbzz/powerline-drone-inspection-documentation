"""End-to-end inspection pipeline (MVP-4.3).

RGB image (+ optional thermal) → YOLO detection → thermal hotspot detection →
FaultRecord(s) → risk scoring → SQLite persistence.

Each RGB detection becomes a record (COMBINED if it overlaps a thermal hotspot,
otherwise EQUIPMENT_DETECTED). Thermal hotspots not overlapping any detection become
standalone HOTSPOT records.

Usage::

    uv run python scripts/run_inspection_pipeline.py --demo
    uv run python scripts/run_inspection_pipeline.py --rgb path/to/img.jpg [--thermal path/to.npy]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
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
    FaultType,
    GpsLocation,
    ThermalData,
)
from src.fault.risk_scorer import apply_risk_to_record  # noqa: E402
from src.thermal.hotspot_detector import detect_hotspots  # noqa: E402

DEFAULT_MODEL = "models/trained/yolov8m_8class_v1/weights/best.pt"
DEFAULT_DB = "outputs/db/faults.db"

CLASS_NAMES = {
    0: "insulator_polymer", 1: "insulator_glass", 2: "insulator_porcelain",
    3: "insulator_generic", 4: "tower", 5: "spacer", 6: "damper", 7: "tower_id_plate",
}


# --------------------------------------------------------------------------- #
# Detection helpers                                                            #
# --------------------------------------------------------------------------- #
def process_rgb_image(model, image_path: str, conf_threshold: float = 0.25) -> list[dict]:
    """Run YOLO on an RGB image; return a list of detection dicts."""
    if image_path is None:
        return []
    results = model.predict(image_path, conf=conf_threshold, verbose=False)
    detections: list[dict] = []
    for r in results:
        if r.boxes is None:
            continue
        img_h, img_w = int(r.orig_shape[0]), int(r.orig_shape[1])
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, str(cls_id)),
                "confidence": conf,
                "bbox": (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                "image_size": (img_w, img_h),
            })
    return detections


def process_thermal_image(thermal_path: Optional[str],
                          target_size: Optional[tuple[int, int]] = None):
    """Load a thermal .npy and detect hotspots. Returns (hotspots, image_stats).

    ``target_size`` (w, h), when given, resizes the radiometric map into the RGB
    image frame so hotspot bboxes can be compared against RGB detections (demo
    pairing only — real deployment needs sensor-aligned RGB/thermal).
    """
    if thermal_path is None or not Path(thermal_path).exists():
        return [], {}
    thermal = np.load(thermal_path).astype(np.float32)
    if target_size is not None:
        thermal = cv2.resize(thermal, target_size, interpolation=cv2.INTER_LINEAR)
    result = detect_hotspots(thermal, method="hybrid")
    return result.hotspots, result.image_stats


def bbox_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    """True if two [x, y, w, h] bboxes intersect (area > 0)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    return ix2 > ix1 and iy2 > iy1


# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
def _thermal_data(hs, ambient: float) -> ThermalData:
    return ThermalData(
        peak_temp_c=hs.peak_temp_c,
        mean_temp_c=hs.mean_temp_c,
        ambient_temp_c=round(ambient, 2),
        area_pixels=hs.area_pixels,
        temperature_delta_c=round(hs.peak_temp_c - ambient, 2),
    )


def run_pipeline(rgb_path, thermal_path, gps, flight_id, model, db,
                 conf_threshold: float = 0.25) -> list[FaultRecord]:
    """Full pipeline for one (rgb, thermal) pair. Either path may be None."""
    label = Path(rgb_path).name if rgb_path else Path(thermal_path).name
    logger.info(f"Processing: {label}")

    detections = process_rgb_image(model, rgb_path, conf_threshold) if rgb_path else []
    logger.info(f"  RGB detections: {len(detections)}")

    # If both RGB & thermal, align thermal into the RGB frame for overlap (demo).
    target_size = None
    if rgb_path and thermal_path:
        if detections:
            target_size = detections[0]["image_size"]
        else:
            with __import__("PIL.Image", fromlist=["Image"]).Image.open(rgb_path) as im:
                target_size = im.size  # (w, h)
    hotspots, thermal_stats = process_thermal_image(thermal_path, target_size)
    if thermal_path:
        logger.info(f"  Thermal hotspots: {len(hotspots)}")

    ambient = float(thermal_stats.get("median", 20.0))
    records: list[FaultRecord] = []

    # One record per RGB detection (COMBINED if it overlaps a hotspot).
    for det in detections:
        matched = next((hs for hs in hotspots if bbox_overlap(det["bbox"], hs.bbox)), None)
        fault_type = FaultType.COMBINED if matched else FaultType.EQUIPMENT_DETECTED
        thermal_data = _thermal_data(matched, ambient) if matched else None
        record = FaultRecord(
            timestamp=datetime.now(),
            gps=gps,
            rgb_image_path=str(rgb_path),
            thermal_image_path=str(thermal_path) if thermal_path else None,
            fault_type=fault_type,
            equipment_class=det["class_name"],
            equipment_class_id=det["class_id"],
            bbox=BoundingBox(*det["bbox"], det["image_size"][0], det["image_size"][1]),
            detection_confidence=det["confidence"],
            thermal=thermal_data,
            inspection_flight_id=flight_id,
        )
        apply_risk_to_record(record)
        records.append(record)

    # Hotspots not matched to any detection → standalone HOTSPOT records.
    for hs in hotspots:
        if any(bbox_overlap(d["bbox"], hs.bbox) for d in detections):
            continue
        record = FaultRecord(
            timestamp=datetime.now(),
            gps=gps,
            rgb_image_path=str(rgb_path) if rgb_path else None,
            thermal_image_path=str(thermal_path),
            fault_type=FaultType.HOTSPOT,
            thermal=_thermal_data(hs, ambient),
            inspection_flight_id=flight_id,
        )
        apply_risk_to_record(record)
        records.append(record)

    for record in records:
        db.create(record)
    logger.info(f"  → {len(records)} fault record(s) created")
    return records


# --------------------------------------------------------------------------- #
# Demo                                                                         #
# --------------------------------------------------------------------------- #
def run_demo(model, db, conf_threshold: float) -> list[FaultRecord]:
    flight_id = "demo_flight_001"
    test_dir = PROJECT_ROOT / "data" / "yolo" / "test" / "images"
    thermal_dir = PROJECT_ROOT / "data" / "synthetic" / "thermal"

    rgb_imgs = sorted(test_dir.glob("*.jpg"))[:5]
    crit = [
        thermal_dir / "05_CriticalHotspot_Joint" / "synth_0501.npy",
        thermal_dir / "06_MultiHotspot" / "synth_0601.npy",
        thermal_dir / "05_CriticalHotspot_Joint" / "synth_0502.npy",
    ]
    if len(rgb_imgs) < 5:
        raise FileNotFoundError(f"Need 5 RGB test images, found {len(rgb_imgs)} in {test_dir}")
    for c in crit:
        if not c.exists():
            raise FileNotFoundError(f"Synthetic thermal missing: {c}")

    base_lat, base_lon = 41.0082, 28.9784  # Istanbul
    # (rgb, thermal) jobs: 2 combined pairs, 3 RGB-only, 1 thermal-only.
    jobs = [
        (rgb_imgs[0], crit[0]),    # combined attempt
        (rgb_imgs[1], crit[1]),    # combined attempt
        (rgb_imgs[2], None),       # RGB-only
        (rgb_imgs[3], None),       # RGB-only
        (rgb_imgs[4], None),       # RGB-only
        (None, crit[2]),           # thermal-only
    ]

    print("=" * 60)
    print("INSPECTION PIPELINE — DEMO RUN")
    print("=" * 60)
    print(f"Flight ID: {flight_id}")
    print("Model: yolov8m_8class_v1")
    print(f"\nProcessing 5 RGB images + 3 thermal maps "
          f"(2 paired/combined, 3 RGB-only, 1 thermal-only)...\n")

    all_records: list[FaultRecord] = []
    for i, (rgb, thermal) in enumerate(jobs, 1):
        gps = GpsLocation(base_lat + i * 0.0007, base_lon + i * 0.0005, altitude_m=80 + i)
        name = (rgb.name if rgb else thermal.name)
        recs = run_pipeline(str(rgb) if rgb else None,
                            str(thermal) if thermal else None,
                            gps, flight_id, model, db, conf_threshold)
        # short per-item summary
        cls = sorted({r.equipment_class for r in recs if r.equipment_class})
        types = sorted({r.fault_type.value for r in recs})
        print(f"[{i}] {name}")
        print(f"    RGB detections: {sum(1 for r in recs if r.bbox is not None)}"
              + (f" ({', '.join(cls)})" if cls else ""))
        print(f"    Thermal: {'yes' if thermal else 'none'} | "
              f"records: {len(recs)} | types: {', '.join(types) if types else '-'}")
        all_records.extend(recs)
    return all_records


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def print_summary(records: list[FaultRecord], db: FaultDatabase, db_path: str) -> None:
    by_risk: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for r in records:
        by_risk[r.risk_level.value] = by_risk.get(r.risk_level.value, 0) + 1
        by_type[r.fault_type.value] = by_type.get(r.fault_type.value, 0) + 1

    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total fault records (this run): {len(records)}")
    print("By risk level:")
    for level in ["low", "medium", "high", "critical"]:
        print(f"  {level:8}: {by_risk.get(level, 0)}")
    print("By fault type:")
    for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {t:20}: {c}")
    print(f"\nDB total records: {db.count()}")
    print(f"DB: {db_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end inspection pipeline.")
    parser.add_argument("--rgb", help="RGB image path", default=None)
    parser.add_argument("--thermal", help="Thermal .npy path (optional)", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--flight-id", default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with InsPLAD test images + synthetic thermal.")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> <level>{level: <8}</level> {message}")

    model_path = (PROJECT_ROOT / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}. Train MVP-2 first (train_yolo.py).")
        sys.exit(1)

    from ultralytics import YOLO
    logger.info(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    db = FaultDatabase(args.db)

    if args.demo:
        records = run_demo(model, db, args.conf)
    else:
        if not args.rgb and not args.thermal:
            logger.error("Provide --rgb and/or --thermal, or use --demo.")
            sys.exit(1)
        if args.rgb and not Path(args.rgb).exists():
            logger.error(f"RGB image not found: {args.rgb}")
            sys.exit(1)
        gps = None
        records = run_pipeline(args.rgb, args.thermal, gps,
                               args.flight_id or "manual_run", model, db, args.conf)

    print_summary(records, db, args.db)


if __name__ == "__main__":
    main()
