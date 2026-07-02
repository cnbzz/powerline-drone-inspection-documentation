"""Industry-standard thermal thresholds for HHATS.

References:
- IEEE C57.106-2015: Guide for Acceptance and Maintenance of Insulating Oil
- IEC 62446-3: Photovoltaic Systems - Outdoor Infrared Thermography
- TEİAŞ İletim Hatları Bakım Yönergesi
"""
from dataclasses import dataclass
from enum import Enum


class ThermalStatus(Enum):
    """Termal durum kategorileri."""
    NORMAL = "normal"
    WARNING = "warning"   # Dikkat
    CRITICAL = "critical"  # Kritik


# === MUTLAK SICAKLIK EŞİKLERİ (peak temperature) ===
# IEC 62446 ve TEİAŞ bakım standartlarına göre
ABSOLUTE_NORMAL_MAX = 60.0       # < 60°C: NORMAL (yeşil)
ABSOLUTE_CRITICAL_MIN = 80.0     # >= 80°C: KRİTİK (kırmızı)
# Aradaki bölge (60-80°C): WARNING (sarı)


# === DELTA SICAKLIK EŞİKLERİ (ΔT = peak - background) ===
# IEEE C57.106 termal yaşlanma standartları
DELTA_NORMAL_MAX = 10.0          # ΔT < 10°C: NORMAL
DELTA_CRITICAL_MIN = 30.0        # ΔT > 30°C: KRİTİK
# Aradaki bölge (10-30°C): WARNING


# === RİSK SKORU KATKILARI ===
RISK_CONTRIBUTION = {
    ThermalStatus.NORMAL: 0,
    ThermalStatus.WARNING: 25,
    ThermalStatus.CRITICAL: 50,
}


# === UI RENK KODLARI (Apple iOS palette) ===
STATUS_COLORS = {
    ThermalStatus.NORMAL: "#34c759",    # Yeşil
    ThermalStatus.WARNING: "#ffcc00",   # Sarı
    ThermalStatus.CRITICAL: "#ff3b30",  # Kırmızı
}


# === UI ETİKETLERİ (Türkçe) ===
STATUS_LABELS_TR = {
    ThermalStatus.NORMAL: "Normal",
    ThermalStatus.WARNING: "Dikkat",
    ThermalStatus.CRITICAL: "Kritik",
}


# === UI EMOJI ===
STATUS_EMOJI = {
    ThermalStatus.NORMAL: "🟢",
    ThermalStatus.WARNING: "🟡",
    ThermalStatus.CRITICAL: "🔴",
}


@dataclass
class ThermalAssessment:
    """Termal değerlendirme sonucu — tüm metadata."""
    peak_temperature: float          # °C
    delta_temperature: float          # °C (peak - background)
    absolute_status: ThermalStatus
    delta_status: ThermalStatus
    final_status: ThermalStatus       # max(absolute, delta)
    risk_contribution: int            # 0/25/50
    color: str
    label_tr: str
    emoji: str


def classify_absolute_temperature(peak_temp: float) -> ThermalStatus:
    """Mutlak sıcaklığa göre durum sınıflandır.

    Args:
        peak_temp: Peak temperature in °C

    Returns:
        ThermalStatus enum
    """
    if peak_temp >= ABSOLUTE_CRITICAL_MIN:
        return ThermalStatus.CRITICAL
    elif peak_temp >= ABSOLUTE_NORMAL_MAX:
        return ThermalStatus.WARNING
    return ThermalStatus.NORMAL


def classify_delta_temperature(delta_temp: float) -> ThermalStatus:
    """Delta sıcaklığa göre durum sınıflandır.

    Args:
        delta_temp: ΔT in °C (positive value)

    Returns:
        ThermalStatus enum
    """
    abs_delta = abs(delta_temp)
    if abs_delta > DELTA_CRITICAL_MIN:
        return ThermalStatus.CRITICAL
    elif abs_delta >= DELTA_NORMAL_MAX:
        return ThermalStatus.WARNING
    return ThermalStatus.NORMAL


def assess_thermal(peak_temp: float, delta_temp: float) -> ThermalAssessment:
    """Mutlak + delta birlikte değerlendir, EN KÖTÜ kazanır.

    Args:
        peak_temp: Peak temperature in °C
        delta_temp: Delta temperature in °C

    Returns:
        ThermalAssessment dataclass
    """
    abs_status = classify_absolute_temperature(peak_temp)
    delta_status = classify_delta_temperature(delta_temp)

    # Final = en yüksek seviye (CRITICAL > WARNING > NORMAL)
    status_priority = {
        ThermalStatus.NORMAL: 0,
        ThermalStatus.WARNING: 1,
        ThermalStatus.CRITICAL: 2,
    }
    final_status = (abs_status if status_priority[abs_status] >= status_priority[delta_status]
                    else delta_status)

    return ThermalAssessment(
        peak_temperature=peak_temp,
        delta_temperature=delta_temp,
        absolute_status=abs_status,
        delta_status=delta_status,
        final_status=final_status,
        risk_contribution=RISK_CONTRIBUTION[final_status],
        color=STATUS_COLORS[final_status],
        label_tr=STATUS_LABELS_TR[final_status],
        emoji=STATUS_EMOJI[final_status],
    )
