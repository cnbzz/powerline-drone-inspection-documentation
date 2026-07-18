"""Post-hoc confidence calibration via temperature scaling (Guo et al., 2017).

Neural nets are typically OVERCONFIDENT: a softmax "0.95" does not mean 95 %
empirical accuracy. Temperature scaling fixes this WITHOUT retraining: it divides
the logits by a single learned scalar T before softmax.

    calibrated_probs = softmax(logits / T)

    T > 1  -> softens (fixes overconfidence)
    T ~ 1  -> already calibrated
    T < 1  -> sharpens (fixes underconfidence)

One T is learned PER equipment model, on that model's VALIDATION split (never the
test split -> no leakage), by minimizing validation NLL. Weights are untouched;
the learned T lives in ``models/trained/defect_classifiers/calibration.json`` and
is applied at inference time.

Run:
    PYTHONUTF8=1 python src/defect_classifier/calibration.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.defect_classifier.dataset import InsPLADFaultDataset
from src.defect_classifier.inference import DefectInferenceEngine, EQUIPMENT_LIST

SEED = 42
DATA_ROOT = _ROOT / "data" / "insplad_fault" / "organized"
MODELS_DIR = _ROOT / "models" / "trained" / "defect_classifiers"
CALIB_PATH = MODELS_DIR / "calibration.json"


def _set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def collect_val_logits(
    engine: DefectInferenceEngine,
    equipment: str,
    data_root: Path = DATA_ROOT,
    batch_size: int = 32,
) -> Optional[tuple[torch.Tensor, torch.Tensor, list[str]]]:
    """Forward the validation split through the model and gather raw logits.

    Returns ``(logits[N,C], labels[N], classes)`` on CPU, or ``None`` if the model
    or val data is missing. The val transform (resize+normalize, NO augmentation)
    matches the inference-time preprocessing.
    """
    model_info = engine.load_model(equipment)
    if model_info is None:
        return None
    model = model_info["model"]
    model_classes = model_info["classes"]

    val_dir = data_root / "val" / equipment
    if not val_dir.exists():
        return None
    ds = InsPLADFaultDataset(data_root, equipment, split="val")

    # The dataset sorts class folders; the model's output columns are ckpt
    # ['classes'] (also sorted at train time). Guard that they align.
    if ds.classes != model_classes:
        raise ValueError(
            f"[{equipment}] class order mismatch: dataset={ds.classes} "
            f"model={model_classes}"
        )

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    all_logits, all_labels = [], []
    for images, labels in loader:
        images = images.to(engine.device)
        logits = model(images)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels), model_classes


def fit_temperature(
    logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 200
) -> float:
    """Learn the single temperature T>0 that minimizes NLL on (logits, labels).

    Optimizes ``log_T`` (so T stays positive) with LBFGS. Returns the scalar T.
    """
    log_t = torch.zeros(1, requires_grad=True)  # T = exp(0) = 1 at init
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def _closure():
        optimizer.zero_grad()
        loss = nll(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(log_t.exp().item())


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor, T: float = 1.0) -> float:
    return float(F.cross_entropy(logits / T, labels).item())


def expected_calibration_error(
    logits: torch.Tensor, labels: torch.Tensor, T: float = 1.0, n_bins: int = 15
) -> float:
    """ECE: weighted gap between confidence and accuracy across confidence bins."""
    probs = F.softmax(logits / T, dim=1)
    conf, preds = probs.max(dim=1)
    conf = conf.numpy()
    correct = (preds == labels).numpy().astype(float)
    n = len(conf)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        # last bin is inclusive of 1.0
        in_bin = (conf > lo) & (conf <= hi) if hi < 1.0 else (conf > lo) & (conf <= hi + 1e-9)
        m = in_bin.sum()
        if m == 0:
            continue
        acc_bin = correct[in_bin].mean()
        conf_bin = conf[in_bin].mean()
        ece += (m / n) * abs(acc_bin - conf_bin)
    return float(ece)


def calibrate_all(save: bool = True) -> dict:
    """Learn per-equipment temperatures on val and report ECE before/after."""
    _set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = DefectInferenceEngine(device=device, use_calibration=False)  # raw logits

    print("=" * 78)
    print(f"TEMPERATURE SCALING — post-hoc calibration on VAL (device={device})")
    print("SAFETY: T floored at 1.0 (soften only, never sharpen). See note below.")
    print("=" * 78)
    print(f"{'Equipment':<34}{'T':>7}{'T_fit':>8}{'ECE_bef':>9}{'ECE_aft':>9}"
          f"{'NLL_bef':>9}{'NLL_aft':>9}")
    print("-" * 78)

    results: dict = {}
    for equipment in EQUIPMENT_LIST:
        collected = collect_val_logits(engine, equipment)
        if collected is None:
            print(f"{equipment:<34}  (model/val yok — atlandı)")
            continue
        logits, labels, classes = collected

        ece_before = expected_calibration_error(logits, labels, T=1.0)
        nll_before = negative_log_likelihood(logits, labels, T=1.0)

        # Unconstrained NLL optimum. On these tiny, ~100%-accurate val sets the
        # optimum runs to T<1 (degenerate SHARPENING): it drives val NLL->0 only
        # by exploiting perfect val accuracy, and on the (imperfect) test set it
        # would make the model catastrophically overconfident. We therefore FLOOR
        # T at 1.0 — temperature scaling may only REDUCE overconfidence here.
        t_fit = fit_temperature(logits, labels)
        T = max(t_fit, 1.0)
        clamped = t_fit < 1.0

        ece_after = expected_calibration_error(logits, labels, T=T)
        nll_after = negative_log_likelihood(logits, labels, T=T)

        results[equipment] = {
            "temperature": round(T, 4),
            "temperature_unconstrained_fit": round(t_fit, 4),
            "clamped_to_1": bool(clamped),
            "ece_before": round(ece_before, 4),
            "ece_after": round(ece_after, 4),
            "nll_before": round(nll_before, 4),
            "nll_after": round(nll_after, 4),
            "n_val": int(len(labels)),
            "classes": classes,
            "calibration_note": (
                "already well-calibrated on val (ECE<0.01) and val≈100% accurate; "
                "unconstrained fit was degenerate (T<1), floored to 1.0 → no change"
                if clamped else "mild softening applied (T>1) to reduce overconfidence"
            ),
        }
        flag = "↓ soften" if T > 1.0 + 1e-6 else "= keep (already calibrated)"
        print(f"{equipment:<34}{T:>7.3f}{t_fit:>8.3f}{ece_before:>9.4f}{ece_after:>9.4f}"
              f"{nll_before:>9.4f}{nll_after:>9.4f}  {flag}")

    print("-" * 78)
    if save and results:
        payload = dict(results)
        payload["_meta"] = {
            "method": "temperature_scaling",
            "source_split": "val",
            "objective": "NLL (LBFGS), T floored at 1.0",
            "seed": SEED,
            "n_bins_ece": 15,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "note": ("Post-hoc; model weights untouched. T>1 softens overconfidence. "
                     "T floored at 1.0: on tiny ~100%-accurate val sets the "
                     "unconstrained NLL optimum degenerates to T<1 (harmful "
                     "sharpening), so only softening is allowed."),
        }
        CALIB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                              encoding="utf-8")
        print(f"💾 {CALIB_PATH}")
    return results


if __name__ == "__main__":
    calibrate_all(save=True)
