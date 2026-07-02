"""LSTM degradation-forecasting model.

OUTPUT CHOICE & RATIONALE
-------------------------
We use **(a) one-step-ahead risk-score regression**: given the last K monthly
feature steps, predict the *next month's* normalized risk score (0-1). Inference
then rolls the model forward autoregressively to obtain (i) the full future risk
curve and (ii) "months to critical" (first month the projected risk crosses 75).

Why (a) over (b) "directly regress months-to-critical":
  * It reuses the existing 0-100 risk score directly as the learning signal, so
    the forecast stays consistent with HHATS's current scoring (risk_scorer.py).
  * A single one-step model yields BOTH the future curve AND remaining-months,
    instead of just a brittle scalar RUL number.
  * Autoregressive forecasting is the standard, defensible formulation and
    degrades gracefully (we can stop the rollout at any horizon).

The model is deliberately tiny (1 LSTM layer, 32 hidden units) — the synthetic
data is small and we want fast, overfit-resistant training on an RTX 4060 8GB.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DegradationLSTM(nn.Module):
    """Small LSTM regressor: (B, K, F) feature window -> (B,) next-step risk (0-1)."""

    def __init__(self, input_size: int = 4, hidden_size: int = 32,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, F)
        out, _ = self.lstm(x)
        last = out[:, -1, :]            # last timestep hidden state
        y = self.head(last).squeeze(-1)  # (B,)
        # Risk is normalized to [0, 1]; keep output in range for stable rollout.
        return torch.sigmoid(y)
