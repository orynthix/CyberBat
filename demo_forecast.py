"""Run one forecast from the local compact cache and trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.dataset.feature_extractor import FEATURE_NAMES
from src.forecasting import forecast_from_models, load_forecast_models, normalize_cached_features


def forecast_from_cache(
    checkpoint_path: str | Path,
    cache_dir: str | Path,
    horizon: int | None = None,
) -> dict[str, object]:
    """Forecast from the latest complete sequence in an existing local cache."""

    cache_path = Path(cache_dir)
    metadata_path = cache_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"cache metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    count = int(metadata["window_count"])
    feature_count = int(metadata["feature_count"])
    if feature_count != len(FEATURE_NAMES):
        raise ValueError(
            f"cache has {feature_count} features; checkpoint expects {len(FEATURE_NAMES)}"
        )

    (
        encoder,
        world_model,
        attack_head,
        config,
        checkpoint_normalizer,
        label_semantics,
        attack_supervision,
    ) = load_forecast_models(checkpoint_path)
    sequence_length = int(config.windows.sequence_length)
    steps = int(horizon if horizon is not None else config.windows.rollout_steps)
    if count < sequence_length:
        raise ValueError(
            f"cache has {count} windows; at least {sequence_length} are required"
        )
    features = np.memmap(
        cache_path / metadata["features_file"],
        dtype=metadata.get("features_dtype", "float32"),
        mode="r",
        shape=(count, feature_count),
    )
    cached_values = np.asarray(features[count - sequence_length : count], dtype=np.float32)
    history = normalize_cached_features(cached_values, metadata, checkpoint_normalizer)
    result = forecast_from_models(history, encoder, world_model, attack_head, steps)
    stage_supported = label_semantics in {"source_stage", "mitre_stage"}
    if not stage_supported:
        result_data = result.to_dict()
        result_data["current_stage_name"] = "Unavailable (checkpoint stage labels unverified)"
        for item in result_data["trajectory"]:
            item["stage_name"] = "Unavailable (checkpoint stage labels unverified)"
    else:
        result_data = result.to_dict()
    return {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "cache": str(cache_path.resolve()),
        "label_semantics": label_semantics,
        "stage_predictions_supported": stage_supported,
        "attack_supervision": attack_supervision,
        "horizon_supervision_valid": attack_supervision
        == "horizon_aligned_rollout_prefixes_v1",
        "temporal_order_verified": (
            metadata.get("windowing", {}).get("timestamp_used") is True
            and metadata.get("windowing", {}).get("timestamp_order_verified") is True
        ),
        "normalization": "cache restored to raw units and transformed with checkpoint training normalizer",
        **result_data,
    }


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
    parser.add_argument("--horizon", type=int, default=None, help="Override the checkpoint K-step horizon")
    args = parser.parse_args()
    print(json.dumps(forecast_from_cache(args.checkpoint, args.cache, args.horizon), indent=2))


if __name__ == "__main__":
    main()
