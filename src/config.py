"""Central configuration for the offline network world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

MITRE_STAGES: Final[dict[int, str]] = {
    0: "Benign",
    1: "Reconnaissance",
    2: "Initial Access",
    3: "Lateral Movement",
    4: "Command and Control",
    5: "Exfiltration",
}

# Dataset-category heuristic used only when a source does not provide ATT&CK
# labels. It is intentionally explicit so it can be replaced with analyst
# reviewed labels without changing the ingestion pipeline.
ATTACK_CATEGORY_TO_STAGE: Final[dict[str, int]] = {
    "normal": 0,
    "benign": 0,
    "background": 0,
    "reconnaissance": 1,
    "analysis": 2,
    "exploits": 2,
    "fuzzers": 2,
    "shellcode": 2,
    "backdoor": 2,
    "dos": 2,
    "worms": 3,
    "generic": 4,
    "botnet": 4,
}


@dataclass(frozen=True)
class WindowConfig:
    """Time-window and sequence settings used by feature extraction."""

    window_seconds: float = 5.0
    stride_seconds: float = 5.0
    sequence_length: int = 12
    rollout_steps: int = 5

    def __post_init__(self) -> None:
        if self.window_seconds <= 0 or self.stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if self.sequence_length < 1 or self.rollout_steps < 1:
            raise ValueError("sequence_length and rollout_steps must be at least 1")


@dataclass(frozen=True)
class ModelConfig:
    """Neural model hyperparameters."""

    latent_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    backbone: str = "transformer"

    def __post_init__(self) -> None:
        if self.latent_dim < 1 or self.hidden_dim < 1:
            raise ValueError("latent_dim and hidden_dim must be positive")
        if self.num_layers < 1 or self.num_heads < 1:
            raise ValueError("num_layers and num_heads must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        if self.backbone not in {"transformer", "lstm"}:
            raise ValueError("backbone must be either 'transformer' or 'lstm'")


@dataclass(frozen=True)
class DataConfig:
    """Dataset locations and reproducibility settings."""

    data_root: Path = field(default_factory=lambda: Path("data"))
    train_path: Path = field(default_factory=lambda: Path("data/train.csv"))
    validation_path: Path = field(default_factory=lambda: Path("data/validation.csv"))
    test_path: Path = field(default_factory=lambda: Path("data/test.csv"))
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")


@dataclass(frozen=True)
class AppConfig:
    """Top-level configuration shared by training and inference."""

    windows: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    learning_rate: float = 1e-3
    batch_size: int = 32
    dynamics_epochs: int = 20
    attack_epochs: int = 20

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.batch_size < 1 or self.dynamics_epochs < 1 or self.attack_epochs < 1:
            raise ValueError("training counts and batch_size must be at least 1")


DEFAULT_CONFIG: Final[AppConfig] = AppConfig()
