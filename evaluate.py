"""Evaluate a trained checkpoint on a separate traffic file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmarks.baseline import (
    calibrate_threshold,
    compare_metrics,
    fit_baseline,
    metrics_from_predictions,
    probability_metrics,
)
from src.dataset.feature_extractor import FEATURE_NAMES, FeatureNormalizer
from src.dataset.large_files import build_compact_cache


def _load_model(
    checkpoint_path: str | Path,
) -> tuple[Any, Any, Any, Any, FeatureNormalizer, str]:
    import torch
    from src.models.attack_head import AttackHead
    from src.models.encoder import StateAutoencoder
    from src.models.world_model import WorldModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    encoder = StateAutoencoder(len(FEATURE_NAMES), config.model.latent_dim, config.model.hidden_dim)
    world_model = WorldModel(
        config.model.latent_dim,
        config.model.hidden_dim,
        config.model.num_layers,
        config.model.num_heads,
        config.model.dropout,
        config.model.backbone,
    )
    attack_head = AttackHead(config.model.latent_dim, config.model.hidden_dim)
    encoder.load_state_dict(checkpoint["encoder"])
    world_model.load_state_dict(checkpoint["world_model"])
    attack_head.load_state_dict(checkpoint["attack_head"])
    encoder.eval()
    world_model.eval()
    attack_head.eval()
    return (
        encoder,
        world_model,
        attack_head,
        config,
        FeatureNormalizer.from_state_dict(checkpoint["normalizer"]),
        checkpoint.get("attack_supervision", "unknown"),
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    train_path: str | Path,
    test_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 512,
    threshold: float = 0.5,
    calibration_samples: int = 2_000,
) -> dict[str, Any]:
    """Evaluate risk predictions and the static baseline on held-out windows."""

    import torch

    output = Path(output_dir)
    if calibration_samples < 1:
        raise ValueError("calibration_samples must be positive")
    output.mkdir(parents=True, exist_ok=True)
    (
        encoder,
        world_model,
        attack_head,
        config,
        checkpoint_normalizer,
        attack_supervision,
    ) = _load_model(checkpoint_path)
    train_cache = build_compact_cache(
        train_path,
        output / "train_cache",
        normalizer=checkpoint_normalizer,
    )
    test_cache = build_compact_cache(
        test_path,
        output / "test_cache",
        normalizer=checkpoint_normalizer,
    )
    count = int(test_cache["window_count"])
    sequence_length = int(config.windows.sequence_length)
    if count <= sequence_length:
        windowing = test_cache.get("windowing", {})
        ordering_note = (
            " The cache uses source-row order without verified timestamps."
            if windowing.get("timestamp_used") is not True
            or windowing.get("timestamp_order_verified") is not True
            else ""
        )
        raise ValueError(
            f"test cache has {count} windows; evaluation requires at least "
            f"{sequence_length + 1} windows for a {sequence_length}-window history "
            f"and its next-window target.{ordering_note}"
        )
    features = np.memmap(output / "test_cache" / test_cache["features_file"], dtype=test_cache["features_dtype"], mode="r", shape=(count, len(FEATURE_NAMES)))
    labels = np.memmap(output / "test_cache" / test_cache["labels_file"], dtype="int8", mode="r", shape=(count,))
    features_array = np.asarray(features, dtype=np.float32)
    inputs = np.stack([features_array[i : i + sequence_length] for i in range(count - sequence_length)])
    targets = np.asarray(labels[sequence_length:], dtype=np.int64)
    risks: list[np.ndarray] = []
    stages: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            latent = encoder.encode(torch.from_numpy(inputs[start : start + batch_size]))
            trajectory = world_model.forward_rollout(latent, config.windows.rollout_steps)
            risk, stage_logits = attack_head(trajectory[:, :1])
            risks.append(risk.numpy())
            stages.append(stage_logits.argmax(dim=-1).numpy())
    risk_values = np.concatenate(risks)
    stage_values = np.concatenate(stages)
    binary_target = (targets > 0).astype(np.int64)
    train_count = int(train_cache["window_count"])
    train_features = np.memmap(
        output / "train_cache" / train_cache["features_file"],
        dtype=train_cache["features_dtype"],
        mode="r",
        shape=(train_count, len(FEATURE_NAMES)),
    )
    train_labels = np.memmap(
        output / "train_cache" / train_cache["labels_file"],
        dtype="int8",
        mode="r",
        shape=(train_count,),
    )
    baseline = fit_baseline(
        np.asarray(train_features, dtype=np.float32),
        (np.asarray(train_labels) > 0).astype(np.int64),
    )
    baseline_metrics = metrics_from_predictions(binary_target, baseline.predict(inputs[:, -1]))
    if test_cache.get("label_semantics") == "binary_attack":
        stage_accuracy = None
        stage_accuracy_note = "unavailable: test cache labels are binary, not MITRE-aligned stage IDs"
    else:
        stage_accuracy = float((stage_values == targets).mean())
        stage_accuracy_note = "compares model stage IDs to source stage labels"
    calibration_count = min(train_count - sequence_length, calibration_samples)
    calibration_indices = np.linspace(0, train_count - sequence_length - 1, calibration_count, dtype=np.int64)
    train_inputs = np.stack(
        [np.asarray(train_features[i : i + sequence_length], dtype=np.float32) for i in calibration_indices]
    )
    train_targets = (np.asarray(train_labels[calibration_indices + sequence_length]) > 0).astype(np.int64)
    with torch.no_grad():
        train_risks: list[np.ndarray] = []
        for start in range(0, len(train_inputs), batch_size):
            latent = encoder.encode(torch.from_numpy(train_inputs[start : start + batch_size]))
            trajectory = world_model.forward_rollout(latent, config.windows.rollout_steps)
            train_risk, _ = attack_head(trajectory[:, :1])
            train_risks.append(train_risk.numpy())
    calibrated_threshold = calibrate_threshold(train_targets, np.concatenate(train_risks))
    world_metrics = probability_metrics(binary_target, risk_values, threshold)
    calibrated_world_metrics = probability_metrics(binary_target, risk_values, calibrated_threshold)
    result = {
        "evaluation_protocol": {
            "schema": "cyberbat.heldout-evaluation.v2",
            "normalization": "checkpoint_training_normalizer_applied_to_train_and_test",
            "attack_supervision": attack_supervision,
            "horizon_supervision_valid": (
                attack_supervision == "horizon_aligned_rollout_prefixes_v1"
            ),
            "split": "train_source_fit_baseline_and_calibration_test_source_evaluation",
            "temporal_order_verified": bool(
                test_cache.get("windowing", {}).get("timestamp_used") is True
                and test_cache.get("windowing", {}).get("timestamp_order_verified") is True
            ),
            "interpretation": (
                "chronological_forecast_evaluation"
                if test_cache.get("windowing", {}).get("timestamp_used") is True
                and test_cache.get("windowing", {}).get("timestamp_order_verified") is True
                else "ordered_window_classification_only_temporal_order_unverified"
            ),
        },
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "test_path": str(Path(test_path).resolve()),
        "train_path": str(Path(train_path).resolve()),
        "samples": int(len(targets)),
        "threshold": threshold,
        "calibration_samples": int(calibration_count),
        "calibrated_threshold": calibrated_threshold,
        "world_model": world_metrics,
        "world_model_calibrated": calibrated_world_metrics,
        "logistic_regression": baseline_metrics,
        "stage_accuracy": stage_accuracy,
        "stage_accuracy_note": stage_accuracy_note,
        "test_windowing": test_cache.get("windowing", {}),
        "train_windowing": train_cache.get("windowing", {}),
        "comparison": compare_metrics(baseline_metrics, world_metrics),
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output", default="artifacts/evaluation")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--calibration-samples", type=int, default=2_000)
    args = parser.parse_args()
    result = evaluate_checkpoint(
        args.checkpoint,
        args.train_data,
        args.test_data,
        args.output,
        args.batch_size,
        args.threshold,
        args.calibration_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
