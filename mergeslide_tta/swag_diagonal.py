"""
SWAG-Diagonal posterior estimation cho MergeSlide vision encoder.

Mục đích: Tính xấp xỉ Gaussian của posterior p(theta | D_source) qua
các SGD iterates SWAG, lưu hai model copies (mean, variance) dùng cho
log q(theta) trong PETAL regularizer tại TTA time.

Cách dùng:
    swag = SWAGDiagonal(vision_encoder)
    for epoch in range(n_swag_epochs):
        train_one_epoch(...)
        swag.update(vision_encoder)
    swag.save("checkpoints/swag_diagonal/fold_0.pt")

    # Tại TTA time:
    mean_sd, var_sd = SWAGDiagonal.load(path)
"""

from __future__ import annotations
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn


class SWAGDiagonal:
    """
    Stochastic Weight Averaging Gaussian (diagonal covariance).

    Theo dõi running mean và running mean-of-squares của vision_encoder
    qua các SGD iterates để xấp xỉ posterior:
        q(theta) = N(mu_SWAG, diag(sigma^2_SWAG))
    với sigma^2_p = E[theta_p^2] - E[theta_p]^2.

    Chỉ track các parameters của vision_encoder (backbone), không track mlp.
    """

    def __init__(self, model: nn.Module):
        self.n_collected = 0

        # Chỉ track LayerNorm parameters
        _ln_names = {
            f"{mod_name}.{param_name}"
            for mod_name, module in model.named_modules()
            if isinstance(module, nn.LayerNorm)
            for param_name, _ in module.named_parameters()
        }

        # running mean E[theta]
        self._mean: dict[str, torch.Tensor] = {
            n: p.data.clone().zero_()
            for n, p in model.named_parameters()
            if n in _ln_names
        }
        # running mean of squares E[theta^2]
        self._sq: dict[str, torch.Tensor] = {
            n: p.data.clone().zero_()
            for n, p in model.named_parameters()
            if n in _ln_names
        }

    def update(self, model: nn.Module) -> None:
        """Cập nhật running statistics từ current model weights."""
        self.n_collected += 1
        alpha = 1.0 / self.n_collected          # equal weighting
        for n, p in model.named_parameters():
            if n in self._mean:
                d = p.data
                self._mean[n].mul_(1.0 - alpha).add_(d, alpha=alpha)
                self._sq[n].mul_(1.0 - alpha).add_(d.pow(2), alpha=alpha)

    def get_variance(self, eps: float = 1e-6) -> dict[str, torch.Tensor]:
        """sigma^2_p = E[theta_p^2] - E[theta_p]^2, clipped to [eps, inf)."""
        return {
            n: torch.clamp(self._sq[n] - self._mean[n].pow(2), min=eps)
            for n in self._mean
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "n_collected": self.n_collected,
            "mean": self._mean,
            "sq":   self._sq,
        }, path)
        print(f"[SWAGDiagonal] Saved {self.n_collected} iterates → {path}")

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> tuple[dict, dict]:
        """
        Load SWAG statistics.

        Returns:
            (mean_state_dict, var_state_dict): cả hai đều là dict mapping
            param_name → Tensor, đã ở trên device được chỉ định.
        """
        ckpt = torch.load(path, map_location=device)
        mean_sd = {n: v.to(device) for n, v in ckpt["mean"].items()}
        sq_sd   = {n: v.to(device) for n, v in ckpt["sq"].items()}
        eps     = 1e-6
        var_sd  = {n: torch.clamp(sq_sd[n] - mean_sd[n].pow(2), min=eps) for n in mean_sd}
        print(f"[SWAGDiagonal] Loaded from {path} ({ckpt['n_collected']} iterates)")
        return mean_sd, var_sd
