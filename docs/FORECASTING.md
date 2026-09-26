# Forecasting and Replay

This document describes the reusable inference and replay APIs in
`src/forecasting.py`. They use the existing encoder, world model, attack head,
checkpoint, and compact dataset cache; they do not retrain or alter those
components.

## Forecast one observed sequence

`forecast_from_models(history, encoder, world_model, attack_head, horizon)`
accepts one sequence of normalized feature rows with shape
`(sequence_length, feature_count)` (or a singleton batch). It returns a
`ForecastResult` containing:

- `current_risk` and `current_stage`, produced by applying the attack head to
  observed latent history;
- one predicted risk, stage, and latent state per future step;
- `horizon`, plus a linear risk trend and its slope per step.

Prefix risks and stages are calculated by applying the existing attack head to
each predicted-trajectory prefix. They are outputs of the existing model, not
separately trained or calibrated per-horizon predictors. The current saved
checkpoint trained the attack head on predicted rollouts, not with a distinct
observed-state objective; its current score is therefore not independently
supervised or calibrated. A sigmoid value is a model score, not a real-world
probability. `to_dict()` provides a JSON-serializable representation.

Current training code supervises each predicted prefix against the label from
the corresponding future window and stores
`attack_supervision: horizon_aligned_rollout_prefixes_v1` in new checkpoints.
The existing local checkpoint predates this correction; its T+2-and-later
outputs and replay metrics are exploratory, not horizon-aligned validation.

## Replay and compare with actual labels

`replay_forecasts(features, stages, encoder, world_model, attack_head, ...)`
walks a sequence deterministically. A replay origin is the last observed
feature index. The first forecast step is compared to the immediately
following label; step K is compared to the label K rows after the origin.
Each comparison includes both indices to make the alignment auditable.

By default, at most 1,000 origins are selected evenly across the valid replay
range. Pass `max_origins=None` to evaluate every origin. This deterministic
subsampling bounds runtime for large caches; the returned replay settings
record the origin selection and count.

Provide sequential stage labels aligned one-for-one with feature rows.
`None`, NaN, and `-1` are accepted as missing labels. Missing labels appear as
JSON `null` and are excluded independently from each horizon's risk and stage
metrics. If there are no known targets at a horizon, metric values are `null`
and sample count is zero; no target or score is inferred. Stages must
otherwise be integers 0 through 5. Passing `stages=None` marks all labels as
unavailable, which is useful for forecast-only replay.

The `forecast_vs_actual` summary reports threshold-based binary risk metrics,
ROC-AUC and PR-AUC where defined, stage accuracy, known sample count, and
missing-label count per horizon. `write_forecast_report(report, path)` writes
stable, sorted JSON and rejects non-standard NaN/Infinity values.

Label meaning is part of the metric contract. `binary_attack` labels can
support binary risk metrics but never six-class stage accuracy. A `stage_id`
label vector can support stage accuracy only when its values are genuine
sequential stage annotations. Cache metadata such as `source_stage` and
`mitre_stage` is normalized by the CLI to the replay API's `stage_id` setting.

## Reproducible local report

The CLI replays a local compact cache with a checkpoint, records SHA-256
digests for the checkpoint and cache inputs, and writes a machine-readable
report:

```powershell
.\.venv\Scripts\python.exe replay_forecast.py `
  --checkpoint artifacts\unsw_nb15_quick\world_model.pt `
  --cache artifacts\unsw_nb15_quick\cache `
  --output artifacts\unsw_nb15_quick\forecast_replay.json `
  --horizon 5 `
  --threshold 0.5 `
  --max-origins 100 `
  --step-seconds 1
```

The default selects 100 evenly spaced origins. Set `--max-origins 0` to use
all eligible origins. Output provenance and replay parameters should be kept
with the report when results are shared. Replay does not establish that row
order is chronology or add timestamps to the data. `step_seconds` is only a
caller-supplied conversion for interpreting already verified sequential
steps.

## Metric interpretation

Only supplied labels are used. The binary target is derived as `stage > 0`.
Precision is undefined when there are no predicted positives; recall is
defined as zero when there are no actual positives, while ROC-AUC and PR-AUC
are `null` when their required classes are unavailable. Stage accuracy and
all risk metrics are computed only over known labels for that horizon.

Forecast-to-actual replay evaluates future labels at corresponding steps. It
does not treat a label from another row, a missing sequential label, or the
current observed stage as a future target. Chronological early-warning metrics
require timestamped, ordered windows, sequential labels with a defined onset,
and forecasts aligned to issuance time. The UNSW-NB15 Parquet files available
in this workspace contain no timestamps; their replay is row-order label
comparison only, and numerical chronological lead time is unavailable.
