"""Synthetic degradation-trajectory generator for predictive maintenance.

WHY SYNTHETIC (academic honesty):
    HHATS has no longitudinal field data (the drone flies once), so we cannot
    observe how a given insulator's risk evolves month over month. To build and
    validate the *architecture*, we generate physically-grounded synthetic
    degradation trajectories. See :data:`src.predictive.SYNTHETIC_DISCLAIMER`.

WHAT A TRAJECTORY IS:
    For one equipment item we simulate a monthly time series of the existing HHATS
    0-100 ``risk_score`` (consistent with src/fault/risk_scorer.py, where >=75 is
    the CRITICAL band) plus the observable features that drive it (peak thermal
    temperature, thermal delta, detection confidence). Degradation is gradual with
    noise, feedback-accelerated (worse condition -> faster decay), and equipment-
    type dependent (polymer insulators corrode faster than towers, etc.). A subset
    of items also receive an abrupt thermal-event acceleration.

LABELS:
    At every timestep we compute "months to critical" = months until ``risk_score``
    first reaches the critical threshold (75). Items that never reach it within the
    horizon are right-censored (label = None).

Everything is deterministic given a seed (thesis reproducibility).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Critical risk threshold — mirrors score_to_risk_level() in risk_scorer.py (>=75 → CRITICAL).
CRITICAL_RISK = 75.0

# Per-timestep observable features fed to the LSTM (order matters).
FEATURE_NAMES: List[str] = ["risk", "peak_temp", "delta_temp", "confidence"]
# Normalization divisors → roughly map each feature to ~[0, 1] for the model.
FEATURE_NORM: Dict[str, float] = {
    "risk": 100.0, "peak_temp": 120.0, "delta_temp": 60.0, "confidence": 1.0,
}

# Sliding-window length (months of history the model sees).
WINDOW_K = 6
# Default trajectory length (months) and number of synthetic equipment items.
DEFAULT_MONTHS = 36
DEFAULT_N_EQUIPMENT = 1200

# Ambient/scene baseline temperature for the thermal mapping (°C).
AMBIENT_C = 25.0

# Equipment-type degradation profile: (base_rate risk-pts/month, feedback accel,
# probability of a sudden thermal event). Insulators (esp. polymer w/ metal
# shackle corrosion) degrade faster than structural items like towers.
EQUIPMENT_PROFILES: Dict[str, Tuple[float, float, float]] = {
    "insulator_polymer":  (1.9, 0.018, 0.30),
    "insulator_glass":    (1.4, 0.014, 0.22),
    "insulator_porcelain": (1.2, 0.012, 0.18),
    "insulator_generic":  (1.5, 0.014, 0.22),
    "tower":              (0.6, 0.006, 0.06),
    "spacer":             (1.0, 0.010, 0.14),
    "damper":             (1.3, 0.012, 0.18),
    "tower_id_plate":     (0.4, 0.004, 0.03),
}
EQUIPMENT_TYPES: List[str] = list(EQUIPMENT_PROFILES.keys())


def risk_to_peak_temp(risk: float) -> float:
    """Physical mapping risk → peak thermal temperature (°C).

    Kept consistent with the thermal thresholds (60/80°C): thermal enters the
    CRITICAL band only at high risk, so thermal is a contributor — not the sole
    driver — of the risk score."""
    return AMBIENT_C + 5.0 + 0.70 * risk


def peak_to_delta(peak_c: float) -> float:
    """ΔT = peak - ambient (mirrors ThermalData.temperature_delta_c)."""
    return max(0.0, peak_c - AMBIENT_C)


def risk_to_confidence(risk: float) -> float:
    """Detection confidence tends to be high and rises slightly as defects grow."""
    return float(np.clip(0.80 + 0.0015 * risk, 0.0, 0.99))


@dataclass
class Trajectory:
    """One synthetic equipment degradation trajectory (monthly)."""
    equipment_type: str
    risk: np.ndarray            # (T,) 0-100 risk score
    peak_temp: np.ndarray       # (T,) °C
    delta_temp: np.ndarray      # (T,) °C
    confidence: np.ndarray      # (T,) 0-1
    months_to_critical: List[Optional[int]] = field(default_factory=list)

    def feature_matrix(self) -> np.ndarray:
        """(T, F) raw feature matrix in FEATURE_NAMES order."""
        return np.stack([self.risk, self.peak_temp, self.delta_temp,
                         self.confidence], axis=1)


def _compute_months_to_critical(risk: np.ndarray) -> List[Optional[int]]:
    """For each t, months until risk first >= CRITICAL (None if never within horizon)."""
    T = len(risk)
    # First index (>= t) where risk crosses critical.
    out: List[Optional[int]] = []
    crit_idxs = np.where(risk >= CRITICAL_RISK)[0]
    for t in range(T):
        future = crit_idxs[crit_idxs >= t]
        out.append(int(future[0] - t) if len(future) else None)
    return out


def generate_trajectory(rng: np.random.Generator,
                        equipment_type: str,
                        months: int = DEFAULT_MONTHS) -> Trajectory:
    """Generate one physically-grounded degradation trajectory."""
    base_rate, accel, event_p = EQUIPMENT_PROFILES[equipment_type]

    risk = np.zeros(months, dtype=np.float64)
    risk[0] = rng.uniform(3.0, 22.0)            # initial (mostly healthy) condition
    # Per-item rate variation (manufacturing/environment spread).
    item_rate = base_rate * rng.uniform(0.7, 1.4)
    # Optional sudden thermal event (e.g., onset of loose/corroded connection).
    event_month = rng.integers(months // 4, months) if rng.random() < event_p else -1

    for t in range(1, months):
        # Feedback acceleration: the worse the condition, the faster the decay.
        increment = item_rate * (1.0 + accel * risk[t - 1])
        # Seasonal-ish + measurement noise.
        increment += rng.normal(0.0, 0.6)
        if t == event_month:
            increment += rng.uniform(8.0, 18.0)  # abrupt acceleration
        risk[t] = float(np.clip(risk[t - 1] + max(0.0, increment), 0.0, 100.0))

    # Derive consistent observable features from risk (+ small noise).
    peak = risk_to_peak_temp(risk) + rng.normal(0.0, 1.5, months)
    peak = np.clip(peak, AMBIENT_C, 150.0)
    delta = np.clip(peak - AMBIENT_C, 0.0, None)
    conf = np.clip(0.80 + 0.0015 * risk + rng.normal(0.0, 0.02, months), 0.0, 0.99)

    traj = Trajectory(
        equipment_type=equipment_type,
        risk=risk, peak_temp=peak, delta_temp=delta, confidence=conf,
    )
    traj.months_to_critical = _compute_months_to_critical(risk)
    return traj


def build_windowed_dataset(
    n_equipment: int = DEFAULT_N_EQUIPMENT,
    months: int = DEFAULT_MONTHS,
    window_k: int = WINDOW_K,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> Dict:
    """Generate trajectories and slice them into supervised one-step-ahead windows.

    Sample = (K consecutive feature steps) -> (next-step risk, normalized 0-1).
    Train/val split is by EQUIPMENT (not by window) to avoid leakage.

    Returns a dict with normalized arrays plus metadata. Deterministic given seed.
    """
    rng = np.random.default_rng(seed)

    # Assign equipment types round-robin-ish but randomized & reproducible.
    types = rng.choice(EQUIPMENT_TYPES, size=n_equipment)
    trajectories = [generate_trajectory(rng, t, months) for t in types]

    # Equipment-level split.
    idx = np.arange(n_equipment)
    rng.shuffle(idx)
    n_val = int(n_equipment * val_fraction)
    val_ids = set(idx[:n_val].tolist())

    norm_vec = np.array([FEATURE_NORM[f] for f in FEATURE_NAMES], dtype=np.float64)

    def windows_for(traj: Trajectory):
        feats = traj.feature_matrix() / norm_vec          # (T, F) normalized
        risk_norm = traj.risk / FEATURE_NORM["risk"]      # target series
        xs, ys = [], []
        for i in range(months - window_k):
            xs.append(feats[i:i + window_k])
            ys.append(risk_norm[i + window_k])
        return xs, ys

    X_tr, y_tr, X_va, y_va = [], [], [], []
    reached_critical = 0
    for eid, traj in enumerate(trajectories):
        if traj.months_to_critical[0] is not None:
            reached_critical += 1
        xs, ys = windows_for(traj)
        if eid in val_ids:
            X_va.extend(xs); y_va.extend(ys)
        else:
            X_tr.extend(xs); y_tr.extend(ys)

    out = {
        "X_train": np.asarray(X_tr, dtype=np.float32),
        "y_train": np.asarray(y_tr, dtype=np.float32),
        "X_val": np.asarray(X_va, dtype=np.float32),
        "y_val": np.asarray(y_va, dtype=np.float32),
        "feature_names": FEATURE_NAMES,
        "feature_norm": FEATURE_NORM,
        "window_k": window_k,
        "critical_risk": CRITICAL_RISK,
        "months": months,
        "n_equipment": n_equipment,
        "n_reached_critical": reached_critical,
        "seed": seed,
        "is_synthetic": True,
        "sample_trajectories": trajectories[:5],
    }
    return out


if __name__ == "__main__":
    from src.predictive import SYNTHETIC_DISCLAIMER

    print("[SENTETİK VERİ]", SYNTHETIC_DISCLAIMER)
    data = build_windowed_dataset(n_equipment=20, seed=7)
    print("X_train", data["X_train"].shape, "y_train", data["y_train"].shape)
    print("X_val", data["X_val"].shape, "y_val", data["y_val"].shape)
    print("features:", data["feature_names"])
    for tr in data["sample_trajectories"][:3]:
        print(f"  {tr.equipment_type:18s} risk[:6]=",
              np.round(tr.risk[:6], 1), "mtc[0]=", tr.months_to_critical[0])
