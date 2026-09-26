"""Multi-task infiltration risk and MITRE stage prediction head."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class AttackHead(nn.Module):
    """Map a predicted latent trajectory to risk and six MITRE stages."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_stages: int = 6) -> None:
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1 or num_stages < 2:
            raise ValueError("invalid attack-head dimensions")
        self.projection = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU())
        self.risk = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.stage = nn.Linear(hidden_dim, num_stages)

    def forward(self, trajectory: Tensor) -> tuple[Tensor, Tensor]:
        if trajectory.ndim != 3:
            raise ValueError("trajectory must have shape (batch, steps, latent_dim)")
        features = self.projection(trajectory)
        pooled = features.max(dim=1).values
        return self.risk(pooled).squeeze(-1), self.stage(pooled)

    def loss(self, trajectory: Tensor, risk_target: Tensor, stage_target: Tensor) -> Tensor:
        risk, stage_logits = self(trajectory)
        return nn.functional.mse_loss(risk, risk_target.float()) + nn.functional.cross_entropy(stage_logits, stage_target.long())
