"""Fault domain: data model, risk scoring, persistence."""

from src.fault.database import FaultDatabase, FaultRecordORM
from src.fault.fault_record import (
    BoundingBox,
    FaultRecord,
    FaultStatus,
    FaultType,
    GpsLocation,
    RiskLevel,
    ThermalData,
)
from src.fault.risk_scorer import (
    RiskBreakdown,
    apply_risk_to_record,
    score_fault,
    score_to_risk_level,
)

__all__ = [
    "FaultRecord",
    "BoundingBox",
    "ThermalData",
    "GpsLocation",
    "FaultType",
    "FaultStatus",
    "RiskLevel",
    "FaultDatabase",
    "FaultRecordORM",
    "RiskBreakdown",
    "score_fault",
    "apply_risk_to_record",
    "score_to_risk_level",
]
