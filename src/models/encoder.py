"""Neural encoder and decoder for normalized traffic state vectors."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class StateAutoencoder(nn.Module):
    """Compress traffic state vectors into latent states and reconstruct them."""

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden_dim: int = 128) -> None:
        super().__init__()
        if input_dim < 1 or latent_dim < 1 or hidden_dim < 1:
            raise ValueError("encoder dimensions must be positive")
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))

    def encode(self, state: Tensor) -> Tensor:
        return self.encoder(state)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder(latent)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        latent = self.encode(state)
        return latent, self.decode(latent)

    def reconstruction_loss(self, state: Tensor) -> Tensor:
        latent, reconstruction = self(state)
        del latent
        return nn.functional.mse_loss(reconstruction, state)
