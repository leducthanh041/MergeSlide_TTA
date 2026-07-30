"""Diagonal SWAG posterior loader used by test-time regularization."""

from pathlib import Path

import torch


class SWAGDiagonal:
    """Load running SWAG mean and diagonal variance by parameter name."""

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> tuple[dict, dict]:
        checkpoint = torch.load(path, map_location=device)
        mean = {name: value.to(device) for name, value in checkpoint["mean"].items()}
        square_mean = {
            name: value.to(device) for name, value in checkpoint["sq"].items()
        }
        variance = {
            name: torch.clamp(
                square_mean[name] - mean[name].pow(2), min=1e-6
            )
            for name in mean
        }
        print(
            f"[SWAGDiagonal] Loaded {path} "
            f"({checkpoint['n_collected']} iterates)"
        )
        return mean, variance
