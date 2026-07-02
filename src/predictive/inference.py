"""Degradation forecasting inference.

``predict_degradation(history)`` takes a per-equipment recent history (risk +
thermal + confidence) and rolls the trained LSTM forward to estimate:
  * months_to_critical  — months until projected risk crosses the critical band (75)
  * future_risk_curve   — projected 0-100 risk for each future month
  * estimated_critical_date — optional, if a start date is given
  * confidence          — derived from the model's validation error

Graceful: if the model file is missing/corrupt, returns ``None`` with a message
instead of crashing.

NOTE: forecasts come from a model trained on SYNTHETIC data (see
:data:`src.predictive.SYNTHETIC_DISCLAIMER`); treat as architectural demonstration,
not validated field prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from src.predictive import SYNTHETIC_DISCLAIMER
from src.predictive.model import DegradationLSTM
from src.predictive.synthetic_data import (
    peak_to_delta,
    risk_to_confidence,
    risk_to_peak_temp,
)

_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = _ROOT / "models" / "trained" / "predictive" / "lstm_degradation.pt"

# Cached (model, config) so repeated calls don't reload from disk.
_CACHE: Optional[tuple] = None

Observation = Union[Dict[str, float], Sequence[float]]


@dataclass
class DegradationForecast:
    """Result of a degradation projection (Türkçe etiketli alanlar UI içindir)."""
    current_risk: float
    months_to_critical: Optional[int]      # None → ufuk içinde kritiğe ulaşmıyor
    estimated_critical_date: Optional[str]  # ISO date or None
    future_risk_curve: List[float]          # 0-100 risk per future month
    horizon_months: int
    confidence: float                       # 0-1
    is_synthetic: bool = True
    disclaimer: str = SYNTHETIC_DISCLAIMER
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "current_risk": round(self.current_risk, 1),
            "months_to_critical": self.months_to_critical,
            "estimated_critical_date": self.estimated_critical_date,
            "future_risk_curve": [round(v, 1) for v in self.future_risk_curve],
            "horizon_months": self.horizon_months,
            "confidence": round(self.confidence, 2),
            "is_synthetic": self.is_synthetic,
            "disclaimer": self.disclaimer,
            "message": self.message,
        }


def _load() -> Optional[tuple]:
    """Load (model, config, metrics) once; return None if unavailable."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not MODEL_PATH.exists():
        return None
    try:
        # torch 2.6+ defaults weights_only=True; our checkpoint stores a config dict.
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        model = DegradationLSTM(
            input_size=cfg["input_size"],
            hidden_size=cfg.get("hidden_size", 32),
            num_layers=cfg.get("num_layers", 1),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _CACHE = (model, cfg, ckpt.get("metrics", {}))
        return _CACHE
    except Exception:
        return None


def _obs_to_features(obs: Observation, feature_names: List[str]) -> np.ndarray:
    """Normalize one observation into the model's feature order.

    Accepts a dict (risk/peak_temp/delta_temp/confidence; missing thermal/conf are
    derived from risk) or a raw sequence already in feature order."""
    if isinstance(obs, dict):
        risk = float(obs.get("risk", obs.get("risk_score", 0.0)) or 0.0)
        peak = obs.get("peak_temp", obs.get("peak_temp_c"))
        delta = obs.get("delta_temp", obs.get("temperature_delta_c"))
        conf = obs.get("confidence", obs.get("detection_confidence"))
        if peak is None:
            peak = risk_to_peak_temp(risk)
        if delta is None:
            delta = peak_to_delta(peak)
        if conf is None:
            conf = risk_to_confidence(risk)
        values = {"risk": risk, "peak_temp": float(peak),
                  "delta_temp": float(delta), "confidence": float(conf)}
        return np.array([values[f] for f in feature_names], dtype=np.float64)
    return np.asarray(obs, dtype=np.float64)


def predict_degradation(
    history: Sequence[Observation],
    start_date: Optional[date] = None,
    max_horizon: int = 60,
) -> Optional[DegradationForecast]:
    """Project future degradation for one equipment item.

    Args:
        history: recent monthly observations (oldest → newest). May be a single
            current snapshot; it is left-padded to the model's window length.
        start_date: optional reference date to compute an estimated critical date.
        max_horizon: how many months to project forward.

    Returns a :class:`DegradationForecast`, or ``None`` if the model is unavailable.
    """
    loaded = _load()
    if loaded is None:
        return None
    model, cfg, metrics = loaded
    feature_names = cfg["feature_names"]
    feature_norm = cfg["feature_norm"]
    window_k = cfg["window_k"]
    critical = cfg["critical_risk"]
    norm_vec = np.array([feature_norm[f] for f in feature_names], dtype=np.float64)
    risk_idx = feature_names.index("risk")

    if not history:
        return None

    # Build the raw feature window; left-pad short histories with the earliest obs.
    raw = [_obs_to_features(o, feature_names) for o in history]
    while len(raw) < window_k:
        raw.insert(0, raw[0].copy())
    window = np.stack(raw[-window_k:], axis=0)        # (K, F) raw
    current_risk = float(window[-1, risk_idx])

    # Autoregressive rollout.
    future_curve: List[float] = []
    win = window.copy()
    with torch.no_grad():
        for _ in range(max_horizon):
            x = torch.from_numpy((win / norm_vec).astype(np.float32)).unsqueeze(0)
            next_risk_norm = float(model(x).item())
            next_risk = float(np.clip(next_risk_norm * feature_norm["risk"], 0.0, 100.0))
            # Forecast should not "heal": degradation is monotone non-decreasing.
            next_risk = max(next_risk, future_curve[-1] if future_curve else current_risk)
            future_curve.append(next_risk)

            # Synthesize the next full feature vector from projected risk.
            next_peak = risk_to_peak_temp(next_risk)
            next_vec = {
                "risk": next_risk,
                "peak_temp": next_peak,
                "delta_temp": peak_to_delta(next_peak),
                "confidence": risk_to_confidence(next_risk),
            }
            next_row = np.array([next_vec[f] for f in feature_names], dtype=np.float64)
            win = np.vstack([win[1:], next_row])

    # Months to critical (1-based: 1 = next month).
    months_to_critical: Optional[int] = None
    for i, r in enumerate(future_curve):
        if r >= critical:
            months_to_critical = i + 1
            break
    # Already critical now.
    if current_risk >= critical:
        months_to_critical = 0

    est_date: Optional[str] = None
    if start_date is not None and months_to_critical is not None:
        m = start_date.month - 1 + months_to_critical
        y = start_date.year + m // 12
        est_date = date(y, m % 12 + 1, 1).isoformat()

    # Confidence from validation MAE (risk points); lower error → higher confidence.
    val_mae = float(metrics.get("best_val_mae_risk_points", 10.0))
    confidence = float(np.clip(1.0 - val_mae / 50.0, 0.3, 0.95))

    if months_to_critical is None:
        msg = (f"Projeksiyon ufku ({max_horizon} ay) içinde kritik eşik (≥{critical:.0f}) "
               "aşılmıyor.")
    elif months_to_critical == 0:
        msg = "Ekipman ZATEN kritik risk bandında."
    else:
        msg = f"Tahmini ~{months_to_critical} ay içinde kritik risk bandına ulaşır."

    return DegradationForecast(
        current_risk=current_risk,
        months_to_critical=months_to_critical,
        estimated_critical_date=est_date,
        future_risk_curve=future_curve,
        horizon_months=max_horizon,
        confidence=confidence,
        message=msg,
    )


if __name__ == "__main__":
    from datetime import date as _date

    # Örnek: orta riskli, hafif termal yükselen bir izolatör geçmişi.
    sample_history = [
        {"risk": 38.0, "peak_temp": 58.0, "confidence": 0.86},
        {"risk": 41.0, "peak_temp": 60.0, "confidence": 0.87},
        {"risk": 45.0, "peak_temp": 63.0, "confidence": 0.88},
    ]
    fc = predict_degradation(sample_history, start_date=_date(2026, 6, 1))
    if fc is None:
        print("Model bulunamadı — önce eğitim çalıştırılmalı.")
    else:
        import json
        print(json.dumps(fc.to_dict(), ensure_ascii=False, indent=2))
