import numpy as np
import pytest

from src.forecasting import (
    forecast_from_models,
    forecast_lead_time,
    k_horizon_metrics,
    normalize_cached_features,
    replay_forecasts,
    resolve_label_semantics,
    write_forecast_report,
)


def test_forecast_exposes_full_horizon_outputs_and_trend() -> None:
    import torch

    from src.models.attack_head import AttackHead
    from src.models.encoder import StateAutoencoder
    from src.models.world_model import WorldModel

    torch.manual_seed(7)
    encoder = StateAutoencoder(input_dim=4, latent_dim=8, hidden_dim=16)
    world_model = WorldModel(
        latent_dim=8,
        hidden_dim=16,
        num_layers=1,
        num_heads=2,
        dropout=0,
    )
    attack_head = AttackHead(latent_dim=8, hidden_dim=16)

    result = forecast_from_models(np.ones((3, 4), dtype=np.float32), encoder, world_model, attack_head, 3)
    serialized = result.to_dict()

    assert result.horizon == 3
    assert len(result.trajectory) == 3
    assert [step.step for step in result.trajectory] == [1, 2, 3]
    assert all(len(step.latent_state) == 8 for step in result.trajectory)
    assert 0.0 <= result.current_risk <= 1.0
    assert result.current_stage in range(6)
    assert result.trend in {"increasing", "decreasing", "stable"}
    assert len(serialized["trajectory"]) == 3


def test_forecast_validates_horizon_and_single_history() -> None:
    import torch

    from src.models.attack_head import AttackHead
    from src.models.encoder import StateAutoencoder
    from src.models.world_model import WorldModel

    encoder = StateAutoencoder(input_dim=2, latent_dim=4, hidden_dim=8)
    world_model = WorldModel(latent_dim=4, hidden_dim=8, num_layers=1, num_heads=2)
    attack_head = AttackHead(latent_dim=4, hidden_dim=8)
    with pytest.raises(ValueError, match="horizon"):
        forecast_from_models(np.ones((2, 2)), encoder, world_model, attack_head, 0)
    with pytest.raises(ValueError, match="history"):
        forecast_from_models(torch.ones((2, 2, 2)), encoder, world_model, attack_head, 1)


def test_k_horizon_metrics_use_only_supplied_targets() -> None:
    actual = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    scores = np.array([[0.1, 0.1], [0.8, 0.7], [0.4, 0.9], [0.6, 0.2]])

    metrics = k_horizon_metrics(actual, scores, threshold=0.5)

    assert [item["horizon"] for item in metrics] == [1, 2]
    assert [item["samples"] for item in metrics] == [4, 4]
    assert metrics[0]["tp"] == 2
    assert metrics[0]["fp"] == 0
    assert metrics[0]["roc_auc"] == 1.0
    assert metrics[1]["tp"] == 2
    assert metrics[1]["fn"] == 0


def test_k_horizon_metrics_mark_undefined_auc_as_missing() -> None:
    metrics = k_horizon_metrics(np.zeros((2, 1)), np.array([[0.1], [0.9]]))

    assert metrics[0]["roc_auc"] is None
    assert metrics[0]["pr_auc"] is None
    assert metrics[0]["recall"] is None
    assert metrics[0]["fpr"] == 0.5


def test_k_horizon_metrics_mark_undefined_rates_as_missing() -> None:
    metrics = k_horizon_metrics(np.ones((2, 1)), np.zeros((2, 1)))

    assert metrics[0]["precision"] is None
    assert metrics[0]["recall"] == 0.0
    assert metrics[0]["f1"] == 0.0
    assert metrics[0]["fpr"] is None


def test_cache_features_use_checkpoint_training_normalizer() -> None:
    from src.dataset.feature_extractor import FEATURE_NAMES, FeatureNormalizer

    cache_normalizer = FeatureNormalizer()
    cache_normalizer.mean_ = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    cache_normalizer.scale_ = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    cache_normalizer.mean_[0] = 100.0
    cache_normalizer.scale_[0] = 10.0
    model_normalizer = FeatureNormalizer()
    model_normalizer.mean_ = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    model_normalizer.scale_ = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    model_normalizer.mean_[0] = 50.0
    model_normalizer.scale_[0] = 5.0
    model_normalizer.mean_[1] = 2.0
    model_normalizer.scale_[1] = 2.0

    model_features = normalize_cached_features(
        np.array([[0.0, 1.0] + [0.0] * (len(FEATURE_NAMES) - 2)], dtype=np.float32),
        {
            "features_normalized": True,
            "normalizer": cache_normalizer.state_dict(),
        },
        model_normalizer,
    )

    assert np.allclose(model_features[0, :2], [10.0, -0.5])


def test_label_semantics_resolve_legacy_binary_caches_conservatively() -> None:
    assert resolve_label_semantics(None, "unknown", np.array([0, 1, 0])) == "binary_attack"
    assert resolve_label_semantics("source_stage", "unknown", np.array([0, 2])) == "stage_id"
    assert resolve_label_semantics(None, None, np.array([-1, -1])) == "unknown"


def test_lead_time_omits_events_without_advance_warning() -> None:
    actual = np.array([0, 0, 1, 1, 0, 0, 1, 0])
    scores = np.array([0.1, 0.8, 0.9, 0.9, 0.1, 0.1, 0.2, 0.9])

    metrics = forecast_lead_time(actual, scores, threshold=0.5, step_seconds=2)

    assert metrics["events"] == 2
    assert metrics["events_with_advance_warning"] == 1
    assert metrics["advance_warning_rate"] == 0.5
    assert metrics["lead_times_seconds"] == [2.0, None]
    assert metrics["mean_lead_time_seconds"] == 2.0


def test_one_step_forecast_lead_time_uses_forecast_origin() -> None:
    actual = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.8, 0.9, 0.9])

    metrics = forecast_lead_time(
        actual,
        scores,
        threshold=0.5,
        step_seconds=1.0,
        forecast_horizon_windows=1,
    )

    assert metrics["lead_times_seconds"] == [2.0]
    assert metrics["forecast_horizon_windows"] == 1


def test_lead_time_without_events_or_alerts_does_not_invent_a_mean() -> None:
    no_events = forecast_lead_time(np.zeros(3), np.zeros(3))
    no_alerts = forecast_lead_time(np.array([0, 1, 1]), np.zeros(3))
    empty = forecast_lead_time(np.array([]), np.array([]))

    assert no_events["events"] == 0
    assert no_events["advance_warning_rate"] is None
    assert no_events["mean_lead_time_seconds"] is None
    assert no_alerts["lead_times_seconds"] == [None]
    assert no_alerts["mean_lead_time_seconds"] is None
    assert empty["events"] == 0
    assert empty["mean_lead_time_seconds"] is None


@pytest.mark.parametrize(
    ("actual", "scores"),
    [
        (np.array([0, 1]), np.array([0.1])),
        (np.array([0, 2]), np.array([0.1, 0.8])),
        (np.array([0, 1]), np.array([0.1, np.nan])),
    ],
)
def test_lead_time_rejects_misaligned_or_invalid_data(actual: np.ndarray, scores: np.ndarray) -> None:
    with pytest.raises(ValueError):
        forecast_lead_time(actual, scores)


def test_replay_aligns_predictions_and_omits_missing_sequential_labels(tmp_path) -> None:
    import torch

    class IdentityEncoder(torch.nn.Module):
        def encode(self, values):
            return values

    class RepeatWorldModel(torch.nn.Module):
        def forward_rollout(self, history, steps):
            return history[:, -1:, :].expand(-1, steps, -1)

    class FixedAttackHead(torch.nn.Module):
        def forward(self, trajectory):
            batch = trajectory.shape[0]
            risk = torch.full((batch,), 0.8, dtype=trajectory.dtype, device=trajectory.device)
            logits = torch.zeros((batch, 6), dtype=trajectory.dtype, device=trajectory.device)
            logits[:, 1] = 1
            return risk, logits

    features = np.arange(8, dtype=np.float32).reshape(-1, 1)
    stages = np.array([0, 0, 1, None, -1, 2, None, 0], dtype=object)
    arguments = (
        features,
        stages,
        IdentityEncoder(),
        RepeatWorldModel(),
        FixedAttackHead(),
    )
    report = replay_forecasts(
        *arguments,
        sequence_length=2,
        horizon=2,
        threshold=0.5,
        max_origins=2,
    )
    repeated = replay_forecasts(
        *arguments,
        sequence_length=2,
        horizon=2,
        threshold=0.5,
        max_origins=2,
    )

    assert report == repeated
    assert report["replay"]["origin_selection"] == "evenly_spaced"
    assert report["replay"]["origins_evaluated"] == 2
    assert [metric["horizon"] for metric in report["forecast_vs_actual"]] == [1, 2]
    first = report["predictions"][0]
    assert first["origin_index"] == 1
    assert [step["target_index"] for step in first["trajectory"]] == [2, 3]
    assert [step["actual_stage"] for step in first["trajectory"]] == [1, None]
    assert [step["actual_risk"] for step in first["trajectory"]] == [1, None]
    assert report["forecast_vs_actual"][1]["missing_labels"] == 1
    assert report["forecast_vs_actual"][1]["samples"] == 1
    assert report["forecast_vs_actual"][1]["roc_auc"] is None

    output = write_forecast_report(report, tmp_path / "nested" / "report.json")
    assert output.is_file()
    assert '"actual_stage": null' in output.read_text(encoding="utf-8")


def test_binary_attack_replay_does_not_claim_stage_accuracy() -> None:
    import torch

    class IdentityEncoder(torch.nn.Module):
        def encode(self, values):
            return values

    class RepeatWorldModel(torch.nn.Module):
        def forward_rollout(self, history, steps):
            return history[:, -1:, :].expand(-1, steps, -1)

    class FixedAttackHead(torch.nn.Module):
        def forward(self, trajectory):
            batch = trajectory.shape[0]
            return torch.full((batch,), 0.8), torch.zeros((batch, 6))

    report = replay_forecasts(
        np.ones((6, 1), dtype=np.float32),
        np.array([0, 0, 1, 0, 1, 1]),
        IdentityEncoder(),
        RepeatWorldModel(),
        FixedAttackHead(),
        sequence_length=2,
        horizon=2,
        label_semantics="binary_attack",
    )

    assert report["replay"]["label_semantics"] == "binary_attack"
    assert report["predictions"][0]["trajectory"][0]["actual_label"] == 1
    assert report["predictions"][0]["trajectory"][0]["actual_risk"] == 1
    assert report["predictions"][0]["trajectory"][0]["actual_stage"] is None
    assert all(row["stage_accuracy"] is None for row in report["forecast_vs_actual"])


def test_replay_without_sequential_labels_does_not_report_metrics() -> None:
    import torch

    class IdentityEncoder(torch.nn.Module):
        def encode(self, values):
            return values

    class RepeatWorldModel(torch.nn.Module):
        def forward_rollout(self, history, steps):
            return history[:, -1:, :].expand(-1, steps, -1)

    class FixedAttackHead(torch.nn.Module):
        def forward(self, trajectory):
            batch = trajectory.shape[0]
            return (
                torch.full((batch,), 0.5),
                torch.zeros((batch, 6)),
            )

    report = replay_forecasts(
        np.ones((5, 1), dtype=np.float32),
        None,
        IdentityEncoder(),
        RepeatWorldModel(),
        FixedAttackHead(),
        sequence_length=2,
        horizon=2,
    )

    assert report["replay"]["origins_evaluated"] == 2
    assert all(
        step["actual_stage"] is None
        for prediction in report["predictions"]
        for step in prediction["trajectory"]
    )
    assert all(metric["samples"] == 0 for metric in report["forecast_vs_actual"])
    assert all(metric["stage_accuracy"] is None for metric in report["forecast_vs_actual"])


def test_replay_validates_sequential_alignment_and_available_context() -> None:
    class Unused:
        pass

    args = (Unused(), Unused(), Unused())
    with pytest.raises(ValueError, match="aligned"):
        replay_forecasts(
            np.ones((5, 2)),
            np.zeros(4),
            *args,
            sequence_length=2,
            horizon=2,
        )
    with pytest.raises(ValueError, match="at least"):
        replay_forecasts(
            np.ones((3, 2)),
            None,
            *args,
            sequence_length=2,
            horizon=2,
        )
