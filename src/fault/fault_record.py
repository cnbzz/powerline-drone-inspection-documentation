"""Fault record data model (MVP-4).

Domain model for a single detected fault, combining visual (RGB) detection, thermal
anomaly, geo-location and risk assessment. Typical usage:

* RGB only      → fill ``bbox`` + ``equipment_class``, leave ``thermal`` None
* Thermal only  → fill ``thermal``, leave ``equipment_class`` None
* Combined      → fill both, set ``fault_type=COMBINED``
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FaultStatus(str, Enum):
    NEW = "new"
    REVIEWED = "reviewed"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


class FaultType(str, Enum):
    """Fault type classification — multiple categories can be combined."""
    HOTSPOT = "thermal_hotspot"                # thermal anomaly only
    VISUAL_DEFECT = "visual_defect"            # visual problem from RGB model
    EQUIPMENT_DETECTED = "equipment_detected"  # normal equipment detection (reference)
    COMBINED = "combined"                       # RGB + thermal combined
    VEGETATION = "vegetation_encroachment"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Image-frame bounding box."""
    x: int
    y: int
    width: int
    height: int
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class ThermalData:
    """Thermal-image-derived measurements for a fault."""
    peak_temp_c: float
    mean_temp_c: float
    ambient_temp_c: float        # scene median
    area_pixels: int
    temperature_delta_c: float   # peak - ambient


@dataclass
class GpsLocation:
    """Geographic coordinate."""
    latitude: float
    longitude: float
    altitude_m: Optional[float] = None


@dataclass
class FaultRecord:
    """A single fault / detection record."""

    # Identifier
    fault_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)

    # Location
    gps: Optional[GpsLocation] = None

    # Source images
    rgb_image_path: Optional[str] = None
    thermal_image_path: Optional[str] = None  # .npy or .jpg

    # Detection details
    fault_type: FaultType = FaultType.UNKNOWN
    equipment_class: Optional[str] = None       # "insulator_polymer", "tower", ...
    equipment_class_id: Optional[int] = None     # 0-7 (8-class YOLO ID)

    # Geometry (RGB image frame)
    bbox: Optional[BoundingBox] = None

    # Confidence (from YOLO model)
    detection_confidence: Optional[float] = None

    # Thermal data (if applicable)
    thermal: Optional[ThermalData] = None

    # Risk assessment
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0                      # 0-100
    risk_factors: dict = field(default_factory=dict)

    # Lifecycle
    status: FaultStatus = FaultStatus.NEW
    notes: Optional[str] = None
    reviewer: Optional[str] = None
    reviewed_at: Optional[datetime] = None

    # Metadata
    inspection_flight_id: Optional[str] = None

    # --------------------------------------------------------------------- #
    # Serialization                                                          #
    # --------------------------------------------------------------------- #
    def to_dict(self) -> dict:
        """JSON-safe dict: datetimes → ISO strings, Enums → their values."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat() if self.timestamp else None
        d["reviewed_at"] = self.reviewed_at.isoformat() if self.reviewed_at else None
        d["fault_type"] = self.fault_type.value
        d["risk_level"] = self.risk_level.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FaultRecord":
        """Rebuild a FaultRecord from a dict produced by :meth:`to_dict`."""
        data = dict(data)  # shallow copy; don't mutate caller's dict

        def _parse_dt(value) -> Optional[datetime]:
            if value is None:
                return None
            return value if isinstance(value, datetime) else datetime.fromisoformat(value)

        gps = data.get("gps")
        thermal = data.get("thermal")
        bbox = data.get("bbox")

        return cls(
            fault_id=data.get("fault_id", str(uuid.uuid4())),
            timestamp=_parse_dt(data.get("timestamp")) or datetime.now(),
            gps=GpsLocation(**gps) if isinstance(gps, dict) else gps,
            rgb_image_path=data.get("rgb_image_path"),
            thermal_image_path=data.get("thermal_image_path"),
            fault_type=FaultType(data.get("fault_type", FaultType.UNKNOWN.value)),
            equipment_class=data.get("equipment_class"),
            equipment_class_id=data.get("equipment_class_id"),
            bbox=BoundingBox(**bbox) if isinstance(bbox, dict) else bbox,
            detection_confidence=data.get("detection_confidence"),
            thermal=ThermalData(**thermal) if isinstance(thermal, dict) else thermal,
            risk_level=RiskLevel(data.get("risk_level", RiskLevel.LOW.value)),
            risk_score=data.get("risk_score", 0.0),
            risk_factors=data.get("risk_factors") or {},
            status=FaultStatus(data.get("status", FaultStatus.NEW.value)),
            notes=data.get("notes"),
            reviewer=data.get("reviewer"),
            reviewed_at=_parse_dt(data.get("reviewed_at")),
            inspection_flight_id=data.get("inspection_flight_id"),
        )
