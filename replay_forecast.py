"""Generate a reproducible JSON replay report from a local model and cache."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.dataset.feature_extractor import FEATURE_NAMES
from src.forecasting import (
    load_forecast_models,
    normalize_cached_features,
    replay_forecasts,
    resolve_label_semantics,
    write_forecast_report,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_replay_report(
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    output_path: str | Path,
    horizon: int | None = None,
    threshold: float = 0.5,
    max_origins: int | None = 100,
    step_seconds: float = 1.0,
) -> dict[str, Any]:
    """Replay the local compact cache and persist the machine-readable report."""

    checkpoint_file = Path(checkpoint_path)
    cache_path = Path(cache_dir)
    metadata_file = cache_path / "metadata.json"
    if not metadata_file.is_file():
        raise FileNotFoundError(f"cache metadata not found: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    count = int(metadata["window_count"])
    feature_count = int(metadata["feature_count"])
    if feature_count != len(FEATURE_NAMES):
        raise ValueError(
            f"cache has {feature_count} features; checkpoint expects {len(FEATURE_NAMES)}"
        )
    feature_file = cache_path / metadata["features_file"]
    label_file = cache_path / metadata["labels_file"]
    (
        encoder,
        world_model,
        attack_head,
        config,
        checkpoint_normalizer,
        checkpoint_label_semantics,
        attack_supervision,
    ) = load_forecast_models(checkpoint_file)
    selected_horizon = int(horizon if horizon is not None else config.windows.rollout_steps)
    features = np.memmap(
        feature_file,
        dtype=metadata.get("features_dtype", "float32"),
        mode="r",
        shape=(count, feature_count),
    )
    feature_values = normalize_cached_features(features, metadata, checkpoint_normalizer)
    stages = np.memmap(label_file, dtype="int8", mode="r", shape=(count,))
    label_semantics = resolve_label_semantics(
        metadata.get("label_semantics"),
        checkpoint_label_semantics,
        stages,
    )
    report = replay_forecasts(
        feature_values,
        np.asarray(stages),
        encoder,
        world_model,
        attack_head,
        sequence_length=int(config.windows.sequence_length),
        horizon=selected_horizon,
        threshold=threshold,
        step_seconds=step_seconds,
        max_origins=max_origins,
        label_semantics=label_semantics,
    )
    windowing = metadata.get("windowing", {})
    verified_temporal_order = (
        windowing.get("timestamp_used") is True
        and windowing.get("timestamp_order_verified") is True
    )
    report["data_assessment"] = {
        "windowing": windowing or {"method": "legacy_unknown"},
        "label_semantics": label_semantics,
        "attack_supervision": attack_supervision,
        "verified_temporal_order": verified_temporal_order,
        "interpretation": (
            "chronological_forecast_evaluation"
            if verified_temporal_order
            else "row_order_label_replay_not_validated_as_temporal_forecasting"
        ),
        "horizon_supervision_valid": attack_supervision
        == "horizon_aligned_rollout_prefixes_v1",
        "normalization": "cache values restored to raw units then transformed with checkpoint training normalizer",
    }
    report["provenance"] = {
        "checkpoint": str(checkpoint_file.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_file),
        "cache": str(cache_path.resolve()),
        "cache_files": {
            "metadata_sha256": _sha256(metadata_file),
            "features_sha256": _sha256(feature_file),
            "labels_sha256": _sha256(label_file),
        },
    }
    report["output"] = str(Path(output_path).resolve())
    write_forecast_report(report, output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="artifacts/unsw_nb15_quick/world_model.pt",
        help="Local trained checkpoint",
    )
    parser.add_argument(
        "--cache",
        default="artifacts/unsw_nb15_quick/cache",
        help="Existing compact cache directory",
    )
    parser.add_argument(
        "--output",
        default="artifacts/unsw_nb15_quick/forecast_replay.json",
        help="JSON report destination",
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-origins",
        type=int,
        default=100,
        help="Maximum deterministically spaced replay origins; use 0 for all",
    )
    parser.add_argument("--step-seconds", type=float, default=1.0)
    args = parser.parse_args()
    max_origins = None if args.max_origins == 0 else args.max_origins
    report = generate_replay_report(
        args.checkpoint,
        args.cache,
        args.output,
        args.horizon,
        args.threshold,
        max_origins,
        args.step_seconds,
    )
    print(
        json.dumps(
            {
                "output": report["output"],
                "origins_evaluated": report["replay"]["origins_evaluated"],
                "forecast_vs_actual": report["forecast_vs_actual"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
