"""Transformer-first temporal latent-state world model."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class _AttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        attended, weights = self.attention(
            values,
            values,
            values,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )
        values = self.norm1(values + self.dropout(attended))
        values = self.norm2(values + self.dropout(self.feed_forward(values)))
        return values, weights


class WorldModel(nn.Module):
    """Predict the next latent state and recursively roll out future states."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        backbone: str = "transformer",
    ) -> None:
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1 or num_layers < 1 or num_heads < 1:
            raise ValueError("world model dimensions must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if backbone not in {"transformer", "lstm"}:
            raise ValueError("backbone must be 'transformer' or 'lstm'")
        self.backbone = backbone
        self.input_projection = nn.Linear(latent_dim, hidden_dim)
        self.last_attention: Tensor | None = None
        if backbone == "transformer":
            self.temporal = nn.ModuleList(
                [_AttentionBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
            )
        else:
            self.temporal = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, history: Tensor) -> Tensor:
        if history.ndim != 3 or history.shape[1] < 1:
            raise ValueError("history must have shape (batch, sequence>=1, latent_dim)")
        projected = self.input_projection(history)
        if self.backbone == "transformer":
            mask = torch.triu(
                torch.ones(
                    projected.shape[1],
                    projected.shape[1],
                    dtype=torch.bool,
                    device=projected.device,
                ),
                diagonal=1,
            )
            attention_weights: list[Tensor] = []
            encoded = projected
            for block in self.temporal:
                encoded, weights = block(encoded, mask)
                attention_weights.append(weights)
            self.last_attention = torch.stack(attention_weights).mean(dim=0)
        else:
            encoded, _ = self.temporal(projected)
            self.last_attention = torch.full(
                (history.shape[0], history.shape[1], history.shape[1]),
                1.0 / history.shape[1],
                device=history.device,
            )
        return self.output_projection(encoded[:, -1])

    def forward_rollout(self, current_state: Tensor, steps: int) -> Tensor:
        """Return `(batch, steps, latent_dim)` recursive predictions."""

        if current_state.ndim != 3:
            raise ValueError("current_state must have shape (batch, sequence, latent_dim)")
        if steps < 1:
            raise ValueError("steps must be at least 1")
        history = current_state
        predictions = []
        for _ in range(steps):
            next_state = self(history)
            predictions.append(next_state)
            history = torch.cat((history[:, 1:], next_state.unsqueeze(1)), dim=1)
        return torch.stack(predictions, dim=1)
