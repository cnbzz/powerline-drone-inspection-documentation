"""Train the LSTM degradation forecaster on synthetic trajectories.

Fast by design (small synthetic data, tiny model) — minutes, not hours. Saves the
best-val model to models/trained/predictive/lstm_degradation.pt and the training
curve/metrics to outputs/training_logs/predictive/.

Run:  PYTHONUTF8=1 uv run python -m src.predictive.train
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.predictive import SYNTHETIC_DISCLAIMER  # noqa: E402
from src.predictive.model import DegradationLSTM  # noqa: E402
from src.predictive.synthetic_data import build_windowed_dataset  # noqa: E402

MODEL_PATH = _ROOT / "models" / "trained" / "predictive" / "lstm_degradation.pt"
LOG_DIR = _ROOT / "outputs" / "training_logs" / "predictive"

SEED = 42
EPOCHS = 60
BATCH_SIZE = 256
LR = 2e-3


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(seed: int = SEED, epochs: int = EPOCHS) -> dict:
    print("=" * 70)
    print("[SENTETİK VERİ UYARISI]", SYNTHETIC_DISCLAIMER)
    print("=" * 70)
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    data = build_windowed_dataset(seed=seed)
    print(f"Sentetik veri: {data['n_equipment']} ekipman × {data['months']} ay  "
          f"(kritiğe ulaşan: {data['n_reached_critical']})")
    print(f"X_train={data['X_train'].shape}  X_val={data['X_val'].shape}  "
          f"özellikler={data['feature_names']}")

    X_tr = torch.from_numpy(data["X_train"])
    y_tr = torch.from_numpy(data["y_train"])
    X_va = torch.from_numpy(data["X_val"]).to(device)
    y_va = torch.from_numpy(data["y_val"]).to(device)

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE,
                              shuffle=True)

    model = DegradationLSTM(input_size=len(data["feature_names"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = loss_fn(val_pred, y_va).item()
            # Mean absolute error back in 0-100 risk points (interpretable).
            val_mae_pts = (val_pred - y_va).abs().mean().item() * 100.0

        tr_loss = float(np.mean(tr_losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  train={tr_loss:.5f}  "
                  f"val={val_loss:.5f}  val_MAE={val_mae_pts:.2f} risk-puanı")

    # Final val MAE with best weights.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_val_mae = (model(X_va) - y_va).abs().mean().item() * 100.0

    # --- Save model + config (academic-honest metadata) ---
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state": best_state,
        "config": {
            "input_size": len(data["feature_names"]),
            "hidden_size": 32,
            "num_layers": 1,
            "window_k": data["window_k"],
            "feature_names": data["feature_names"],
            "feature_norm": data["feature_norm"],
            "critical_risk": data["critical_risk"],
            "months": data["months"],
        },
        "metrics": {
            "best_val_loss": best_val,
            "best_val_mae_risk_points": best_val_mae,
            "epochs": epochs,
        },
        "is_synthetic": True,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "output": "one-step-ahead normalized risk (0-1); roll forward for curve + months-to-critical",
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"\nModel kaydedildi: {MODEL_PATH}")
    print(f"En iyi val_loss={best_val:.5f}  val_MAE={best_val_mae:.2f} risk-puanı")

    # --- Save training log ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = {
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "is_synthetic": True,
        "seed": seed,
        "epochs": epochs,
        "device": str(device),
        "best_val_loss": best_val,
        "best_val_mae_risk_points": best_val_mae,
        "history": history,
        "data_shapes": {
            "X_train": list(data["X_train"].shape),
            "X_val": list(data["X_val"].shape),
        },
    }
    (LOG_DIR / "training_log.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Eğitim logu: {LOG_DIR / 'training_log.json'}")

    return {"best_val_loss": best_val, "best_val_mae": best_val_mae,
            "first_val": history["val_loss"][0], "last_val": history["val_loss"][-1]}


if __name__ == "__main__":
    train()
