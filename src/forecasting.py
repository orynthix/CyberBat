"""Reusable forecasting outputs and evaluation metrics."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.config import MITRE_STAGES
from src.dataset.feature_extractor import FeatureNormalizer

FORECAST_REPORT_SCHEMA = "cyberbat.forecast-replay.v1"


@dataclass(frozen=True)
class ForecastStep:
    """Model outputs at one prefix of the predicted latent trajectory."""

    step: int
    risk: float
    stage: int
    stage_name: str
    latent_state: tuple[float, ...]


@dataclass(frozen=True)
class ForecastResult:
    """Serializable, single-sample summary of a K-step model rollout."""

    current_risk: float
    current_stage: int
    current_stage_name: str
    horizon: int
    trend: str
    trend_slope_per_step: float
    trajectory: tuple[ForecastStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def forecast_from_models(
    history: Any,
    encoder: Any,
    world_model: Any,
    attack_head: Any,
    horizon: int,
    trend_tolerance: float = 1e-6,
) -> ForecastResult:
    """Encode one feature history and summarize its K-step model rollout.

    Prefix risk and stage values are produced by applying the existing attack
    head to each available prefix of the actual predicted latent trajectory.
    ``current_risk`` and ``current_stage`` are computed from the observed
    latent history, while ``trajectory`` contains only future steps.
    """

    import torch

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not np.isfinite(trend_tolerance) or trend_tolerance < 0:
        raise ValueError("trend_tolerance must be finite and non-negative")
    try:
        device = next(encoder.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    values = torch.as_tensor(history, dtype=torch.float32, device=device)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] < 1 or values.shape[2] < 1:
        raise ValueError("history must have shape (sequence, features) or (1, sequence, features)")
    if not torch.isfinite(values).all():
        raise ValueError("history must contain only finite values")

    encoder.eval()
    world_model.eval()
    attack_head.eval()
    with torch.no_grad():
        latent_history = encoder.encode(values)
        current_risk_tensor, current_stage_logits = attack_head(latent_history)
        predicted_latents = world_model.forward_rollout(latent_history, horizon)
        if not torch.isfinite(predicted_latents).all():
            raise RuntimeError("world model produced non-finite forecast states")
        steps: list[ForecastStep] = []
        for index in range(horizon):
            risk, stage_logits = attack_head(predicted_latents[:, : index + 1])
            if not torch.isfinite(risk).all() or not torch.isfinite(stage_logits).all():
                raise RuntimeError("attack head produced non-finite forecast outputs")
            stage = int(stage_logits[0].argmax().item())
            latent_state = tuple(float(value) for value in predicted_latents[0, index].cpu().tolist())
            steps.append(
                ForecastStep(
                    step=index + 1,
                    risk=float(risk[0].item()),
                    stage=stage,
                    stage_name=MITRE_STAGES.get(stage, f"Unknown ({stage})"),
                    latent_state=latent_state,
                )
            )

    risk_values = np.asarray([step.risk for step in steps], dtype=np.float64)
    slope = (
        float(np.polyfit(np.arange(horizon, dtype=np.float64), risk_values, 1)[0])
        if horizon > 1
        else 0.0
    )
    trend = "increasing" if slope > trend_tolerance else "decreasing" if slope < -trend_tolerance else "stable"
    current_stage = int(current_stage_logits[0].argmax().item())
    return ForecastResult(
        current_risk=float(current_risk_tensor[0].item()),
        current_stage=current_stage,
        current_stage_name=MITRE_STAGES.get(current_stage, f"Unknown ({current_stage})"),
        horizon=horizon,
        trend=trend,
        trend_slope_per_step=slope,
        trajectory=tuple(steps),
    )


def load_forecast_models(
    checkpoint_path: str | Path,
) -> tuple[Any, Any, Any, Any, FeatureNormalizer, str, str]:
    """Load model components, normalization and training-label semantics."""

    import torch

    from src.dataset.feature_extractor import FEATURE_NAMES
    from src.models.attack_head import AttackHead
    from src.models.encoder import StateAutoencoder
    from src.models.world_model import WorldModel

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    encoder = StateAutoencoder(
        len(FEATURE_NAMES), config.model.latent_dim, config.model.hidden_dim
    )
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
    normalizer = FeatureNormalizer.from_state_dict(checkpoint["normalizer"])
    return (
        encoder,
        world_model,
        attack_head,
        config,
        normalizer,
        checkpoint.get("label_semantics", "unknown"),
        checkpoint.get("attack_supervision", "unknown"),
    )


def normalize_cached_features(
    features: np.ndarray,
    cache_metadata: dict[str, Any],
    checkpoint_normalizer: FeatureNormalizer,
) -> np.ndarray:
    """Transform cache values into the feature scale expected by a checkpoint.

    Compact caches store normalized float16 values. They are first restored to
    raw feature units using their own saved normalizer, then transformed with
    the model's training normalizer.
    """

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("cached features must be a 2-D matrix")
    if cache_metadata.get("features_normalized", False):
        cache_normalizer = FeatureNormalizer.from_state_dict(cache_metadata["normalizer"])
        values = cache_normalizer.inverse_transform(values)
    return checkpoint_normalizer.transform(values)


def resolve_label_semantics(
    cache_semantics: str | None,
    checkpoint_semantics: str | None,
    labels: np.ndarray,
) -> str:
    """Resolve replay target meaning without treating binary labels as stages."""

    for semantics in (cache_semantics, checkpoint_semantics):
        if semantics in {"binary_attack", "stage_id"}:
            return semantics
        if semantics in {"source_stage", "mitre_stage"}:
            return "stage_id"
    values = np.asarray(labels)
    known = values[np.isfinite(values) & (values >= 0)]
    if known.size == 0:
        return "unknown"
    unique_labels = set(np.unique(known).tolist())
    if unique_labels.issubset({0, 1}):
        return "binary_attack"
    if unique_labels.issubset(set(MITRE_STAGES)):
        return "stage_id"
    return "unknown"


def replay_forecasts(
    features: np.ndarray,
    stages: np.ndarray | None,
    encoder: Any,
    world_model: Any,
    attack_head: Any,
    sequence_length: int,
    horizon: int,
    threshold: float = 0.5,
    step_seconds: float = 1.0,
    max_origins: int | None = 1_000,
    label_semantics: str = "stage_id",
) -> dict[str, Any]:
    """Replay forecasts at deterministic origins and compare with future labels.

    A forecast origin is the final observed index in its input history; horizon
    step 1 is compared with the next row. ``None``, NaN, and ``-1`` stage
    labels represent missing sequential labels. Missing targets are kept as
    ``None`` in the report and excluded from that horizon's metrics.
    """

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("features must be a non-empty 2-D array")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")
    if sequence_length < 1:
        raise ValueError("sequence_length must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    if max_origins is not None and max_origins < 1:
        raise ValueError("max_origins must be positive or None")
    if label_semantics not in {"stage_id", "binary_attack", "unknown"}:
        raise ValueError("label_semantics must be stage_id, binary_attack, or unknown")

    if stages is None:
        stage_values: list[int | None] = [None] * len(values)
    else:
        raw_stages = np.asarray(stages, dtype=object)
        if raw_stages.ndim != 1 or len(raw_stages) != len(values):
            raise ValueError("stages must be a 1-D array aligned with features")
        stage_values = []
        for value in raw_stages:
            if value is None:
                stage_values.append(None)
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError("stages must contain integer MITRE stages or missing values") from exc
            if np.isnan(parsed) or parsed == -1:
                stage_values.append(None)
            elif not np.isfinite(parsed) or not parsed.is_integer() or int(parsed) not in MITRE_STAGES:
                raise ValueError("stages must be MITRE stage integers 0-5, -1, or missing")
            else:
                stage_values.append(int(parsed))

    first_origin = sequence_length - 1
    last_origin = len(values) - horizon - 1
    if last_origin < first_origin:
        raise ValueError(
            "replay requires at least sequence_length + horizon feature rows"
        )
    available_origins = np.arange(first_origin, last_origin + 1, dtype=np.int64)
    if max_origins is not None and len(available_origins) > max_origins:
        origin_indices = np.linspace(
            0, len(available_origins) - 1, num=max_origins, dtype=np.int64
        )
        available_origins = available_origins[origin_indices]

    predictions: list[dict[str, Any]] = []
    for origin in available_origins:
        forecast = forecast_from_models(
            values[origin - sequence_length + 1 : origin + 1],
            encoder,
            world_model,
            attack_head,
            horizon,
        )
        predicted_steps: list[dict[str, Any]] = []
        for step in forecast.trajectory:
            target_index = int(origin + step.step)
            actual_stage = stage_values[target_index]
            predicted_steps.append(
                {
                    "horizon": step.step,
                    "target_index": target_index,
                    "predicted_risk": step.risk,
                    "predicted_stage": step.stage,
                    "predicted_stage_name": step.stage_name,
                    "actual_label": actual_stage,
                    "actual_risk": (
                        None
                        if actual_stage is None or label_semantics == "unknown"
                        else int(actual_stage > 0)
                    ),
                    "actual_stage": actual_stage if label_semantics == "stage_id" else None,
                }
            )
        predictions.append(
            {
                "origin_index": int(origin),
                "current_risk": forecast.current_risk,
                "current_stage": forecast.current_stage,
                "current_stage_name": forecast.current_stage_name,
                "trend": forecast.trend,
                "trend_slope_per_step": forecast.trend_slope_per_step,
                "trajectory": predicted_steps,
            }
        )

    metrics: list[dict[str, Any]] = []
    for step_index in range(horizon):
        known_risk = [
            (prediction["trajectory"][step_index], prediction["trajectory"][step_index]["actual_risk"])
            for prediction in predictions
            if prediction["trajectory"][step_index]["actual_risk"] is not None
        ]
        known_stage = [
            (prediction["trajectory"][step_index], prediction["trajectory"][step_index]["actual_stage"])
            for prediction in predictions
            if prediction["trajectory"][step_index]["actual_stage"] is not None
        ]
        if known_risk:
            binary_metrics = k_horizon_metrics(
                np.asarray([[int(label)] for _, label in known_risk], dtype=np.int64),
                np.asarray([[entry["predicted_risk"]] for entry, _ in known_risk], dtype=np.float64),
                threshold,
            )[0]
        else:
            binary_metrics = {
                "horizon": step_index + 1,
                "samples": 0,
                "f1": None,
                "precision": None,
                "recall": None,
                "fpr": None,
                "roc_auc": None,
                "pr_auc": None,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "tp": 0,
            }
        stage_accuracy: float | None = (
            float(np.mean([entry["predicted_stage"] == stage for entry, stage in known_stage]))
            if known_stage
            else None
        )
        metrics.append(
            {
                **binary_metrics,
                "horizon": step_index + 1,
                "stage_accuracy": stage_accuracy,
                "stage_samples": len(known_stage),
                "missing_labels": len(predictions) - len(known_risk),
            }
        )

    return {
        "schema": FORECAST_REPORT_SCHEMA,
        "replay": {
            "sequence_length": int(sequence_length),
            "horizon": int(horizon),
            "threshold": float(threshold),
            "step_seconds": float(step_seconds),
            "origin_selection": "all" if max_origins is None else "evenly_spaced",
            "max_origins": max_origins,
            "origins_evaluated": len(predictions),
            "label_semantics": label_semantics,
        },
        "predictions": predictions,
        "forecast_vs_actual": metrics,
    }


def write_forecast_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write a stable JSON replay report, rejecting non-standard NaN values."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def k_horizon_metrics(
    actual: np.ndarray,
    predicted_scores: np.ndarray,
    threshold: float = 0.5,
) -> list[dict[str, float | int | None]]:
    """Evaluate binary risk scores independently at each supplied horizon.

    Inputs are aligned ``(samples, horizons)`` arrays. Only supplied targets
    are evaluated; undefined ROC-AUC/PR-AUC values are returned as ``None``.
    """

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError as exc:
        raise RuntimeError("forecast metrics require scikit-learn") from exc

    labels = np.asarray(actual)
    scores = np.asarray(predicted_scores, dtype=np.float64)
    if labels.ndim != 2 or scores.shape != labels.shape or labels.shape[0] < 1 or labels.shape[1] < 1:
        raise ValueError("actual and predicted_scores must be matching non-empty 2-D arrays")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("actual labels must contain only 0 or 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("predicted scores must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")

    results: list[dict[str, float | int | None]] = []
    for index in range(labels.shape[1]):
        horizon_labels = labels[:, index].astype(np.int64)
        horizon_scores = scores[:, index]
        predictions = horizon_scores >= threshold
        positives = horizon_labels == 1
        negatives = ~positives
        true_positive = int(np.count_nonzero(predictions & positives))
        false_positive = int(np.count_nonzero(predictions & negatives))
        false_negative = int(np.count_nonzero(~predictions & positives))
        true_negative = int(np.count_nonzero(~predictions & negatives))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1_denominator = 2 * true_positive + false_positive + false_negative
        f1 = (
            2 * true_positive / f1_denominator
            if f1_denominator
            else None
        )
        results.append(
            {
                "horizon": index + 1,
                "samples": int(len(horizon_labels)),
                "f1": float(f1) if f1 is not None else None,
                "precision": float(precision) if precision is not None else None,
                "recall": float(recall) if recall is not None else None,
                "fpr": float(false_positive / (false_positive + true_negative))
                if false_positive + true_negative
                else None,
                "roc_auc": float(roc_auc_score(horizon_labels, horizon_scores))
                if positives.any() and negatives.any()
                else None,
                "pr_auc": float(average_precision_score(horizon_labels, horizon_scores))
                if positives.any()
                else None,
                "tn": true_negative,
                "fp": false_positive,
                "fn": false_negative,
                "tp": true_positive,
            }
        )
    return results


def forecast_lead_time(
    actual: np.ndarray,
    predicted_scores: np.ndarray,
    threshold: float = 0.5,
    step_seconds: float = 1.0,
    forecast_horizon_windows: int = 0,
) -> dict[str, Any]:
    """Measure alerts preceding actual attack onsets in an aligned time series.

    Each contiguous positive run is one event. Scores are aligned to their
    target-label index; ``forecast_horizon_windows`` converts that target index
    back to the time when the forecast was issued. Events without a prior alert
    retain a ``None`` lead time and do not enter the mean.
    """

    labels = np.asarray(actual)
    scores = np.asarray(predicted_scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape:
        raise ValueError("actual and predicted_scores must be matching 1-D arrays")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("actual labels must contain only 0 or 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("predicted scores must be finite values in [0, 1]")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if not np.isfinite(step_seconds) or step_seconds <= 0:
        raise ValueError("step_seconds must be finite and positive")
    if forecast_horizon_windows < 0:
        raise ValueError("forecast_horizon_windows cannot be negative")

    previous_labels = np.r_[False, labels[:-1] == 1]
    onsets = np.flatnonzero((labels == 1) & ~previous_labels)
    alert_indices = np.flatnonzero(scores >= threshold)
    lead_times: list[float | None] = []
    previous_event_end = -1
    for onset in onsets:
        prior_alerts = alert_indices[
            (alert_indices > previous_event_end) & (alert_indices < onset)
        ]
        lead_windows = (
            onset - (int(prior_alerts[-1]) - forecast_horizon_windows)
            if len(prior_alerts)
            else 0
        )
        lead_times.append(float(lead_windows * step_seconds) if len(prior_alerts) else None)
        previous_event_end = int(np.flatnonzero(labels[onset:] == 0)[0] + onset - 1) if (labels[onset:] == 0).any() else len(labels) - 1

    observed = [value for value in lead_times if value is not None]
    return {
        "events": int(len(onsets)),
        "events_with_advance_warning": len(observed),
        "advance_warning_rate": len(observed) / len(onsets) if len(onsets) else None,
        "lead_times_seconds": lead_times,
        "mean_lead_time_seconds": float(np.mean(observed)) if observed else None,
        "step_seconds": float(step_seconds),
        "forecast_horizon_windows": int(forecast_horizon_windows),
    }
