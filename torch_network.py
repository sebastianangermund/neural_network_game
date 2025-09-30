# torch_network.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TorchNetConfig:
    input_dim: int
    hidden1: int = 32
    hidden2: int = 32
    lr: float = 1e-3
    device: Optional[str] = None   # "cuda" | "cpu" | None -> auto


class _MLP(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_dim)

        # Kaiming init for hidden, small init for output
        for m in (self.fc1, self.fc2):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.uniform_(self.out.weight, -0.01, 0.01)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # tanh keeps outputs in [-1, 1], fits “direction” target
        return torch.tanh(self.out(x))


class TorchSteeringNet:
    """
    Minimal PyTorch wrapper with a drop-in API:
      - .train_step(x_np, y_np) -> np.array shape (2,)
      - .predict(x_np) -> np.array shape (2,)
      - .save(path), .load(path)

    Expects x_np shape (input_dim, 1), y_np shape (2, 1).
    """
    def __init__(self, cfg: TorchNetConfig):
        dev = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)
        self.model = _MLP(cfg.input_dim, cfg.hidden1, cfg.hidden2).to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr.ravel()
        t = torch.from_numpy(arr.astype(np.float32))
        return t.to(self.device)

    @torch.no_grad()
    def predict(self, x_np: np.ndarray) -> np.ndarray:
        x = self._to_tensor(x_np).unsqueeze(0)  # [1, D]
        y = self.model(x)                       # [1, 2]
        return y.squeeze(0).detach().cpu().numpy()

    def train_step(self, x_np: np.ndarray, y_np: np.ndarray) -> np.ndarray:
        """
        One online SGD step with MSE loss.
        Returns the raw prediction (np.array of shape (2,)).
        """
        self.model.train()
        x = self._to_tensor(x_np).unsqueeze(0)  # [1, D]
        y = self._to_tensor(y_np).unsqueeze(0)  # [1, 2]

        pred = self.model(x)                    # [1, 2]

        # Normalize BOTH target and prediction to unit vectors for stability
        pred_norm = torch.clamp(pred.norm(p=2, dim=-1, keepdim=True), min=1e-6)
        y_norm = torch.clamp(y.norm(p=2, dim=-1, keepdim=True), min=1e-6)
        pred_unit = pred / pred_norm
        y_unit = y / y_norm

        loss = F.mse_loss(pred_unit, y_unit)

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()

        return pred.detach().squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        sd = torch.load(path, map_location=self.device)
        self.model.load_state_dict(sd)
