"""
inference_lstm.py
─────────────────
Run:  python inference_lstm.py --ckpt lstm_epoch_50.pth --device cpu
"""

import argparse, torch
from torch import nn

# ─── 1.  Re-create *exactly* the same architecture ────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):                               # x → (B, T, F)
        out, _ = self.lstm(x)                           # (B, T, H)
        return self.head(out[:, -1])                    # last step → (B, O)


def load_model(ckpt_path: str, device: str = "cpu"):
    cfg = dict(input_dim=15, hidden_dim=32, output_dim=1)   # ← match training
    model = LSTMModel(**cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state if isinstance(state, dict) else state["model_state"])
    model.eval()                                            # inference mode
    return model


def run(model: nn.Module, batch: torch.Tensor):
    logits = model(batch)           # (B, O)
    return logits