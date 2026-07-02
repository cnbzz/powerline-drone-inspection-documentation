"""Risk scoring engine (MVP-4.2, dynamic-weight normalization).

Weighted 0-100 risk score::

    Risk = 0.30·FaultTypeSeverity + 0.25·ThermalAnomaly + 0.20·Distance/Safety
         + 0.15·DetectionConfidence + 0.10·HistoryRepeat

A component that is *inactive* (no data, or not yet implemented) returns ``None`` and
is dropped from the formula; the remaining weights are renormalized to sum to 1.0,
preserving their relative proportions. This keeps the full 0-100 range reachable even
while Distance and History are placeholders (always ``None`` for now).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.fault.fault_record import FaultRecord, FaultType, RiskLevel, ThermalData
from src.thermal.thresholds import ThermalStatus, assess_thermal

# Component base weights (sum = 1.0). Renormalized over active components at scoring time.
WEIGHTS = {
    "fault_type": 0.30,
    "thermal": 0.25,
    "distance": 0.20,    # activates in MVP-5
    "confidence": 0.15,
    "history": 0.10,     # activates later
}
COMPONENT_ORDER = ["fault_type", "thermal", "distance", "confidence", "history"]

# Fault type severity (0-100)
FAULT_TYPE_SEVERITY = {
    FaultType.HOTSPOT: 70,
    FaultType.VISUAL_DEFECT: 60,
    FaultType.COMBINED: 90,
    FaultType.VEGETATION: 50,
    FaultType.EQUIPMENT_DETECTED: 5,
    FaultType.UNKNOWN: 30,
}


@dataclass
class RiskBreakdown:
    """Detailed account of how a risk score was computed.

    ``raw_scores`` may contain ``None`` for inactive components. ``weights_used`` holds
    the renormalized weight actually applied (0.0 for inactive components).
    """

    raw_scores: dict[str, Optional[float]]
    weights_used: dict[str, float]
    contributions: dict[str, float]
    active_components: list[str]
    total_score: float
    risk_level: RiskLevel

    def to_dict(self) -> dict:
        components = {}
        for k in COMPONENT_ORDER:
            raw = self.raw_scores.get(k)
            active = raw is not None
            components[k] = {
                "raw": round(raw, 2) if active else None,
                "weight_used": round(self.weights_used.get(k, 0.0), 4),
                "contribution": round(self.contributions.get(k, 0.0), 2),
                "active": active,
            }
        return {
            "components": components,
            "active_components": list(self.active_components),
            "total_score": round(self.total_score, 2),
            "risk_level": self.risk_level.value,
        }


# --------------------------------------------------------------------------- #
# Component scorers — return None when inactive / no data                      #
# --------------------------------------------------------------------------- #
def compute_fault_type_score(fault_type: FaultType) -> Optional[float]:
    """0-100 severity from lookup. Always active (fault_type always present)."""
    return float(FAULT_TYPE_SEVERITY.get(fault_type, 30))


# Termal kategori → 0-100 skor (sabit endüstri eşikleri, bkz src/thermal/thresholds.py)
# Bu skor WEIGHTS['thermal']=0.25 ile çarpılır → katkı: Normal +0, Dikkat +6.25, Kritik +12.5
_THERMAL_CATEGORY_SCORE = {
    ThermalStatus.NORMAL: 0.0,
    ThermalStatus.WARNING: 50.0,
    ThermalStatus.CRITICAL: 100.0,
}


def compute_thermal_score(thermal: Optional[ThermalData]) -> Optional[float]:
    """0-100 thermal anomaly score from fixed industry thresholds (60/80°C, ΔT 10/30°C).

    Returns ``None`` when there is no thermal data. Kategori (Normal/Dikkat/Kritik)
    mutlak peak + ΔT'nin EN KÖTÜsünden belirlenir (bkz. assess_thermal)."""
    if thermal is None:
        return None

    assessment = assess_thermal(thermal.peak_temp_c, thermal.temperature_delta_c)
    return _THERMAL_CATEGORY_SCORE[assessment.final_status]


def compute_confidence_score(confidence: Optional[float]) -> Optional[float]:
    """Detection confidence (0-1) → 0-100, or ``None`` when no detection confidence."""
    if confidence is None:
        return None
    return confidence * 100.0


def compute_distance_score(record: FaultRecord) -> Optional[float]:
    """Distance/safety score. Inactive placeholder (``None``) until MVP-5."""
    return None


def compute_history_score(record: FaultRecord) -> Optional[float]:
    """History repeat score. Inactive placeholder (``None``) until implemented."""
    return None


def score_to_risk_level(score: float) -> RiskLevel:
    """Map a 0-100 score to a risk band."""
    if score < 25:
        return RiskLevel.LOW
    if score < 50:
        return RiskLevel.MEDIUM
    if score < 75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL


def score_fault(record: FaultRecord) -> RiskBreakdown:
    """Compute the dynamically-normalized risk breakdown for a :class:`FaultRecord`."""
    raw_scores: dict[str, Optional[float]] = {
        "fault_type": compute_fault_type_score(record.fault_type),
        "thermal": compute_thermal_score(record.thermal),
        "distance": compute_distance_score(record),
        "confidence": compute_confidence_score(record.detection_confidence),
        "history": compute_history_score(record),
    }

    active = {k: v for k, v in raw_scores.items() if v is not None}
    active_weight_sum = sum(WEIGHTS[k] for k in active)

    weights_used: dict[str, float] = {k: 0.0 for k in WEIGHTS}
    contributions: dict[str, float] = {k: 0.0 for k in WEIGHTS}
    total = 0.0

    if active_weight_sum > 0:
        for k, v in active.items():
            w = WEIGHTS[k] / active_weight_sum
            weights_used[k] = w
            contributions[k] = w * v
            total += contributions[k]

    return RiskBreakdown(
        raw_scores=raw_scores,
        weights_used=weights_used,
        contributions=contributions,
        active_components=[k for k in COMPONENT_ORDER if k in active],
        total_score=total,
        risk_level=score_to_risk_level(total),
    )


def apply_risk_to_record(record: FaultRecord) -> FaultRecord:
    """Compute risk and populate the record's risk fields (in place)."""
    breakdown = score_fault(record)
    record.risk_score = round(breakdown.total_score, 2)
    record.risk_level = breakdown.risk_level
    record.risk_factors = breakdown.to_dict()
    return record


# --------------------------------------------------------------------------- #
# Açıklanabilirlik (Stage 7.1A) — UI şeffaflık paneli için                     #
# Risk hesaplama mantığını DEĞİŞTİRMEZ; sadece mevcut breakdown'ı insan-okur   #
# bileşenlere çevirir. Kaynak: record.risk_factors (gösterilen skorla birebir  #
# tutarlı); yoksa score_fault() ile yeniden hesaplanır.                        #
# --------------------------------------------------------------------------- #
COMPONENT_NAMES_TR = {
    "fault_type": "Arıza Tipi",
    "thermal": "Termal Şiddeti",
    "distance": "Mesafe / Güvenlik",
    "confidence": "Tespit Güveni",
    "history": "Geçmiş Tekrar",
}


@dataclass
class RiskComponent:
    """Tek bir risk bileşeninin açıklaması (UI için)."""
    name: str           # "Arıza Tipi"
    name_short: str     # "fault_type"
    weight: float       # uygulanan (renormalize) ağırlık, 0-1
    raw_score: float    # 0-100 ham bileşen skoru
    contribution: float # raw * weight = toplam skora katkı (puan)
    explanation: str    # neden bu skor


@dataclass
class RiskExplanation:
    """Risk skorunun TAM açıklaması (mevcut RiskBreakdown'dan AYRI, UI odaklı)."""
    total_score: float
    risk_level: str
    components: list  # list[RiskComponent]
    threshold_critical: float = 75.0
    threshold_high: float = 50.0
    threshold_medium: float = 25.0  # score_to_risk_level ile birebir (medium >= 25)

    def get_dominant_component(self) -> Optional[RiskComponent]:
        """En yüksek katkı yapan bileşen (yoksa None)."""
        return max(self.components, key=lambda c: c.contribution) if self.components else None


def _explain_fault_type(fault_type, score: float) -> str:
    """Arıza tipi neden bu skoru aldı? (FaultType enum value'larına göre)."""
    val = fault_type.value if hasattr(fault_type, "value") else str(fault_type)
    return {
        "thermal_hotspot": "Termal anomali — yüksek riskli arıza tipi",
        "visual_defect": "Görsel hasar — orta-yüksek riskli",
        "combined": "Birleşik (görsel + termal) hasar — kritik seviye",
        "vegetation_encroachment": "Bitki örtüsü teması — orta riskli",
        "equipment_detected": "Standart ekipman tespiti — düşük riskli",
        "unknown": "Belirsiz arıza tipi",
    }.get(val, f"Arıza tipi: {val} (şiddet {score:.0f}/100)")


def _explain_thermal(thermal) -> str:
    """Termal neden bu skoru aldı? (60/80°C, ΔT 10/30°C eşikleri)."""
    if thermal is None:
        return "Termal veri yok"
    peak = thermal.peak_temp_c
    delta = thermal.temperature_delta_c
    if peak >= 80:
        return f"Peak {peak:.1f}°C → KRİTİK eşik (80°C) aşıldı"
    if peak >= 60:
        return f"Peak {peak:.1f}°C → DİKKAT eşiği (60°C) üstünde"
    if delta > 30:
        return f"Çevreden Δ{delta:.1f}°C fark → anormal ısınma (Kritik)"
    if delta >= 10:
        return f"Δ{delta:.1f}°C fark → orta seviye ısınma (Dikkat)"
    return f"{peak:.1f}°C / Δ{delta:.1f}°C → normal sıcaklık aralığında"


def _explain_confidence(conf) -> str:
    """Tespit güveni açıklaması."""
    if conf is None:
        return "Tespit güveni verisi yok"
    if conf >= 0.9:
        return f"%{conf*100:.0f} yüksek güven → çok güvenilir tespit"
    if conf >= 0.7:
        return f"%{conf*100:.0f} orta güven → güvenilir tespit"
    return f"%{conf*100:.0f} düşük güven → ek doğrulama önerilir"


def explain_risk_score(record: FaultRecord) -> RiskExplanation:
    """Bir arıza için risk skorunun insan-okur açıklaması.

    Öncelik record.risk_factors (kayıtlı, gösterilen skorla tutarlı); yoksa
    score_fault() ile yeniden hesaplanır. Yalnızca AKTİF bileşenler döndürülür.
    """
    rf = getattr(record, "risk_factors", None)
    if not (isinstance(rf, dict) and rf.get("components")):
        rf = score_fault(record).to_dict()

    comps_raw = rf.get("components", {})
    explain_fns = {
        "fault_type": lambda c: _explain_fault_type(record.fault_type, c.get("raw") or 0.0),
        "thermal": lambda c: _explain_thermal(record.thermal),
        "confidence": lambda c: _explain_confidence(record.detection_confidence),
        "distance": lambda c: "Mesafe/güvenlik bileşeni (henüz aktif değil)",
        "history": lambda c: "Geçmiş tekrar bileşeni (henüz aktif değil)",
    }

    components: list = []
    for key in COMPONENT_ORDER:
        c = comps_raw.get(key)
        if not c or not c.get("active"):
            continue
        components.append(RiskComponent(
            name=COMPONENT_NAMES_TR.get(key, key),
            name_short=key,
            weight=float(c.get("weight_used", 0.0)),
            raw_score=float(c.get("raw") or 0.0),
            contribution=float(c.get("contribution", 0.0)),
            explanation=explain_fns.get(key, lambda c: "")(c),
        ))

    total = float(rf.get("total_score", getattr(record, "risk_score", 0.0) or 0.0))
    level = rf.get("risk_level")
    if not level:
        rl = getattr(record, "risk_level", None)
        level = rl.value if hasattr(rl, "value") else (rl or "medium")

    return RiskExplanation(total_score=total, risk_level=level, components=components)
