"""Two-phase offline training entry point."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG
from src.dataset.feature_extractor import FEATURE_NAMES, FeatureNormalizer
from src.dataset.large_files import build_compact_cache


def pretrain_dynamics(
    encoder: Any,
    world_model: Any,
    batches: Iterable[tuple[Any, Any]],
    optimizer: Any,
    epochs: int,
) -> list[float]:
    """Minimize latent next-state MSE over temporal batches."""

    import torch

    history: list[float] = []
    for _ in range(epochs):
        total = 0.0
        count = 0
        for batch in batches:
            states, next_states = batch[:2]
            optimizer.zero_grad(set_to_none=True)
            latent = encoder.encode(states)
            prediction = world_model(latent)
            loss = torch.nn.functional.mse_loss(encoder.decode(prediction), next_states[:, 0])
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("dynamics training requires at least one batch")
        history.append(total / max(count, 1))
    return history


def finetune_attack(
    world_model: Any,
    attack_head: Any,
    batches: Iterable[tuple[Any, Any, Any]],
    optimizer: Any,
    epochs: int,
    rollout_steps: int,
) -> list[float]:
    """Minimize risk regression plus MITRE stage cross-entropy."""

    import torch

    history: list[float] = []
    for _ in range(epochs):
        total = 0.0
        count = 0
        for batch in batches:
            history_states, risk_targets, stage_targets = batch
            if (
                risk_targets.ndim != 2
                or stage_targets.ndim != 2
                or risk_targets.shape != stage_targets.shape
                or risk_targets.shape[1] != rollout_steps
            ):
                raise ValueError(
                    "attack targets must have shape (batch, rollout_steps) with labels aligned to each horizon"
                )
            optimizer.zero_grad(set_to_none=True)
            trajectory = world_model.forward_rollout(history_states, rollout_steps)
            horizon_losses = [
                attack_head.loss(
                    trajectory[:, :step + 1],
                    risk_targets[:, step],
                    stage_targets[:, step],
                )
                for step in range(rollout_steps)
            ]
            loss = torch.stack(horizon_losses).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            count += 1
        if not count:
            raise ValueError("attack training requires at least one batch")
        history.append(total / max(count, 1))
    return history


def train_from_file(
    data_path: str | Path,
    output_dir: str | Path = "artifacts",
    dynamics_epochs: int | None = None,
    attack_epochs: int | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Train and save an end-to-end model from labeled flow/PCAP data.

    Input rows must contain ``risk`` and/or ``stage`` labels. Missing risk is
    derived from stage, while missing stage is treated as benign only when risk
    is also zero.
    """

    try:
        import torch
        from torch.utils.data import DataLoader
        from src.models.attack_head import AttackHead
        from src.models.encoder import StateAutoencoder
        from src.models.world_model import WorldModel
    except ImportError as exc:
        raise RuntimeError("training requires PyTorch; install requirements.txt") from exc

    output = Path(output_dir)
    dynamics_epochs = dynamics_epochs or DEFAULT_CONFIG.dynamics_epochs
    attack_epochs = attack_epochs or DEFAULT_CONFIG.attack_epochs
    batch_size = batch_size or DEFAULT_CONFIG.batch_size
    if dynamics_epochs < 1 or attack_epochs < 1 or batch_size < 1:
        raise ValueError("epoch and batch-size overrides must be positive")
    cache_dir = output / "cache"
    metadata = build_compact_cache(data_path, cache_dir)
    window_count = int(metadata["window_count"])
    sequence_length = DEFAULT_CONFIG.windows.sequence_length
    rollout_steps = DEFAULT_CONFIG.windows.rollout_steps
    if window_count < sequence_length + rollout_steps:
        raise ValueError(
            f"training requires at least {sequence_length + rollout_steps} windows "
            f"for a {sequence_length}-window history and {rollout_steps} labeled future steps; "
            f"found {window_count}"
        )
    features = np.memmap(
        cache_dir / metadata["features_file"],
        dtype=metadata.get("features_dtype", "float32"),
        mode="r",
        shape=(window_count, len(FEATURE_NAMES)),
    )
    labels = np.memmap(cache_dir / metadata["labels_file"], dtype="int8", mode="r", shape=(window_count,))
    normalizer = FeatureNormalizer.from_state_dict(metadata["normalizer"])

    class CachedSequences(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return window_count - sequence_length - rollout_steps + 1

        def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any]:
            current = np.asarray(
                features[index : index + sequence_length + rollout_steps],
                dtype=np.float32,
            )
            if not metadata.get("features_normalized", False):
                current = normalizer.transform(current)
            history = torch.from_numpy(current[:sequence_length].copy())
            future = torch.from_numpy(current[sequence_length:].copy())
            future_stages = torch.from_numpy(
                np.asarray(
                    labels[index + sequence_length : index + sequence_length + rollout_steps],
                    dtype=np.int64,
                ).copy()
            )
            future_risks = (future_stages > 0).to(dtype=torch.float32)
            return history, future, future_risks, future_stages

    dataset = CachedSequences()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = StateAutoencoder(len(FEATURE_NAMES), DEFAULT_CONFIG.model.latent_dim, DEFAULT_CONFIG.model.hidden_dim)
    world_model = WorldModel(
        DEFAULT_CONFIG.model.latent_dim,
        DEFAULT_CONFIG.model.hidden_dim,
        DEFAULT_CONFIG.model.num_layers,
        DEFAULT_CONFIG.model.num_heads,
        DEFAULT_CONFIG.model.dropout,
        DEFAULT_CONFIG.model.backbone,
    )
    attack_head = AttackHead(DEFAULT_CONFIG.model.latent_dim, DEFAULT_CONFIG.model.hidden_dim)
    dynamics_optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()), lr=DEFAULT_CONFIG.learning_rate
    )
    dynamics_history = pretrain_dynamics(
        encoder, world_model, loader, dynamics_optimizer, dynamics_epochs
    )
    attack_optimizer = torch.optim.Adam(
        list(world_model.parameters()) + list(attack_head.parameters()), lr=DEFAULT_CONFIG.learning_rate
    )
    class AttackBatches:
        def __iter__(self):
            for states, _, risks, stages in loader:
                yield encoder.encode(states).detach(), risks, stages

    attack_history = finetune_attack(
        world_model,
        attack_head,
        AttackBatches(),
        attack_optimizer,
        attack_epochs,
        rollout_steps,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "attack_head": attack_head.state_dict(),
        "normalizer": normalizer.state_dict(),
        "feature_names": FEATURE_NAMES,
        "config": DEFAULT_CONFIG,
        "label_semantics": metadata.get("label_semantics", "unknown"),
        "attack_supervision": "horizon_aligned_rollout_prefixes_v1",
        "dynamics_loss": dynamics_history,
        "attack_loss": attack_history,
    }
    checkpoint_path = output / "world_model.pt"
    torch.save(checkpoint, checkpoint_path)
    return {"checkpoint": checkpoint_path, "dynamics_loss": dynamics_history, "attack_loss": attack_history}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the offline network world model")
    parser.add_argument("--data", required=True, help="Path to a flow CSV or PCAP file")
    parser.add_argument("--output", default="artifacts", help="Checkpoint output directory")
    parser.add_argument("--dynamics-epochs", type=int, default=None)
    parser.add_argument("--attack-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    try:
        result = train_from_file(
            args.data,
            args.output,
            dynamics_epochs=args.dynamics_epochs,
            attack_epochs=args.attack_epochs,
            batch_size=args.batch_size,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Saved checkpoint to {result['checkpoint']}")


if __name__ == "__main__":
    main()
