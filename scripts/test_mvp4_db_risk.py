"""Smoke test for MVP-4: Fault Database + Risk Scoring.

Creates 10 representative fault records, scores them, persists to a temporary
SQLite DB, then exercises stats / filters / status-update / risk-breakdown.

Usage::

    uv run python scripts/test_mvp4_db_risk.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

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
    RiskLevel,
    ThermalData,
)
from src.fault.risk_scorer import apply_risk_to_record, score_fault  # noqa: E402


def create_test_records() -> list[FaultRecord]:
    """10 records spanning the fault-type / severity spectrum."""
    records: list[FaultRecord] = []
    base_time = datetime.now()
    base_lat, base_lon = 41.0, 28.9  # Istanbul

    # 1: Visual detection, normal equipment (low risk)
    records.append(FaultRecord(
        timestamp=base_time - timedelta(minutes=10),
        gps=GpsLocation(base_lat, base_lon + 0.001, altitude_m=85),
        fault_type=FaultType.EQUIPMENT_DETECTED,
        equipment_class="insulator_polymer", equipment_class_id=0,
        bbox=BoundingBox(100, 200, 80, 30, 640, 480),
        detection_confidence=0.92,
    ))
    # 2: Visual defect
    records.append(FaultRecord(
        gps=GpsLocation(base_lat, base_lon + 0.002),
        fault_type=FaultType.VISUAL_DEFECT,
        equipment_class="insulator_glass", equipment_class_id=1,
        bbox=BoundingBox(150, 220, 70, 35, 640, 480),
        detection_confidence=0.85, notes="Possible damage detected",
    ))
    # 3: Mild thermal hotspot
    records.append(FaultRecord(
        gps=GpsLocation(base_lat, base_lon + 0.003),
        fault_type=FaultType.HOTSPOT,
        thermal=ThermalData(52.0, 48.0, 22.0, 400, 30.0),
    ))
    # 4: Medium thermal hotspot
    records.append(FaultRecord(
        gps=GpsLocation(base_lat, base_lon + 0.004),
        fault_type=FaultType.HOTSPOT,
        thermal=ThermalData(78.0, 72.0, 20.0, 600, 58.0),
    ))
    # 5: Critical hotspot
    records.append(FaultRecord(
        gps=GpsLocation(base_lat, base_lon + 0.005),
        fault_type=FaultType.HOTSPOT,
        thermal=ThermalData(125.0, 115.0, 20.0, 1200, 105.0),
    ))
    # 6: COMBINED — visual defect + hotspot
    records.append(FaultRecord(
        gps=GpsLocation(base_lat, base_lon + 0.006),
        fault_type=FaultType.COMBINED,
        equipment_class="insulator_polymer", equipment_class_id=0,
        bbox=BoundingBox(200, 250, 80, 40, 640, 480),
        detection_confidence=0.88,
        thermal=ThermalData(85.0, 78.0, 22.0, 500, 63.0),
    ))
    # 7: Vegetation
    records.append(FaultRecord(
        gps=GpsLocation(base_lat + 0.001, base_lon),
        fault_type=FaultType.VEGETATION, equipment_class="tree",
        bbox=BoundingBox(300, 100, 200, 300, 640, 480),
        detection_confidence=0.78,
    ))
    # 8: Low-end hotspot
    records.append(FaultRecord(
        gps=GpsLocation(base_lat + 0.002, base_lon),
        fault_type=FaultType.HOTSPOT,
        thermal=ThermalData(45.0, 42.0, 18.0, 300, 27.0),
    ))
    # 9: Visual defect, low confidence
    records.append(FaultRecord(
        gps=GpsLocation(base_lat + 0.003, base_lon),
        fault_type=FaultType.VISUAL_DEFECT,
        equipment_class="damper", equipment_class_id=6,
        bbox=BoundingBox(400, 300, 50, 25, 640, 480),
        detection_confidence=0.65,
    ))
    # 10: COMBINED, extreme
    records.append(FaultRecord(
        gps=GpsLocation(base_lat + 0.004, base_lon),
        fault_type=FaultType.COMBINED,
        equipment_class="insulator_porcelain", equipment_class_id=2,
        bbox=BoundingBox(250, 280, 90, 35, 640, 480),
        detection_confidence=0.91,
        thermal=ThermalData(140.0, 125.0, 21.0, 1500, 119.0),
    ))
    return records


def main() -> None:
    test_db_path = PROJECT_ROOT / "outputs" / "db" / "test_faults.db"
    test_db_path.unlink(missing_ok=True)
    db = FaultDatabase(test_db_path)

    print("=" * 60)
    print("MVP-4 TEST: Fault Database + Risk Scoring")
    print("=" * 60)

    test_records = create_test_records()
    print(f"\n--- Creating {len(test_records)} test records ---")
    for i, record in enumerate(test_records, 1):
        apply_risk_to_record(record)
        fault_id = db.create(record)
        print(f"{i:2}. [{record.risk_level.value:8}] score={record.risk_score:5.1f} "
              f"type={record.fault_type.value:25} id={fault_id[:8]}")

    print("\n--- DB Stats ---")
    stats = db.stats()
    print(f"Total records: {stats['total']}")
    print("By risk level:")
    for level, count in stats["by_risk_level"].items():
        print(f"  {level:10}: {count}")
    print("By status:")
    for status, count in stats["by_status"].items():
        print(f"  {status:15}: {count}")

    print("\n--- Filter Tests ---")
    high_risk = db.list_all(risk_level=RiskLevel.HIGH)
    critical = db.list_all(risk_level=RiskLevel.CRITICAL)
    print(f"High risk records: {len(high_risk)}")
    print(f"Critical risk records: {len(critical)}")

    if critical:
        first_critical = critical[0]
        db.update_status(first_critical.fault_id, FaultStatus.REVIEWED,
                         reviewer="test_user", notes="Inspected on site")
        updated = db.get(first_critical.fault_id)
        print(f"\nUpdate test: {first_critical.fault_id[:8]} "
              f"status={updated.status.value}, reviewer={updated.reviewer}")

    print("\n--- Risk Breakdown Example (highest-scoring record) ---")
    all_records = db.list_all()
    if all_records:
        # Highest-scoring record overall (critical band may be empty by design).
        record = max(all_records, key=lambda r: r.risk_score)
        breakdown = score_fault(record)
        print(f"Fault: {record.fault_type.value}")
        if record.thermal:
            print(f"  Thermal: peak={record.thermal.peak_temp_c}°C, "
                  f"delta={record.thermal.temperature_delta_c}°C")
        print(f"  Confidence: {record.detection_confidence}")
        print("Score Breakdown (active components renormalized):")
        for component, vals in breakdown.to_dict()["components"].items():
            if vals["active"]:
                print(f"  {component:12}: raw={vals['raw']:6.1f}  "
                      f"weight={vals['weight_used']:.3f} → contribution={vals['contribution']:6.2f}")
            else:
                print(f"  {component:12}: inactive (excluded from formula)")
        print(f"  TOTAL: {breakdown.total_score:.2f} → {breakdown.risk_level.value.upper()}")

    # Round-trip serialization sanity (to_dict / from_dict)
    print("\n--- Serialization round-trip ---")
    sample = test_records[5]
    rebuilt = FaultRecord.from_dict(sample.to_dict())
    ok = (rebuilt.fault_id == sample.fault_id
          and rebuilt.fault_type == sample.fault_type
          and rebuilt.thermal.peak_temp_c == sample.thermal.peak_temp_c
          and rebuilt.risk_level == sample.risk_level)
    print(f"to_dict/from_dict round-trip preserved key fields: {ok}")

    print("\n" + "=" * 60)
    print(f"Test complete. DB at: {test_db_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
