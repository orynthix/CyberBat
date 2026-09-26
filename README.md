# CyberBat — SIH26153 Network Attack Forecasting

CyberBat is an offline temporal network-risk forecasting prototype. It encodes
traffic windows into latent states, learns temporal state transitions, and
recursively simulates future latent states to produce a risk-score trajectory
and MITRE-aligned stage predictions for analyst review. Unlike a conventional
single-window classifier, its central output is **what the model scores for
future states T+1 through T+K**, not merely a label for traffic already seen.

It is a research/demo prototype, not an operational intrusion-prevention
system. A model risk score is not a calibrated real-world probability, and a
forecast does not establish that a future attack will occur.

See [ARCHITECTURE.md](ARCHITECTURE.md) for a compact system diagram and
component view.

## What is implemented

| Capability | Implementation |
|---|---|
| Traffic input | CSV/TSV, Parquet, `.pcap` and `.pcapng` adapters; optional Scapy is used for packet captures. |
| Large-file preparation | PyArrow `ParquetFile.iter_batches`; a Parquet-specific fast path projects recognized columns and aggregates arrays without one Python event per row. Other input paths stream events into online window accumulators. |
| Temporal features | 24 flow/packet-oriented statistics, aggregated into timestamp windows for generic event streams or fixed row groups for the Parquet fast path. |
| Compact data | Normalized `float16` feature matrix, `int8` window labels, JSON metadata/report; features can be memory-mapped. |
| Latent encoder | `StateAutoencoder` maps each 24-value window through a dense encoder to a latent vector. |
| State transitions | Causal Transformer by default; an LSTM option is implemented. |
| Recursive forecast | `WorldModel.forward_rollout()` repeatedly predicts one latent state and appends it to a rolling latent history. |
| Forecast output | Current risk/stage plus prefix risk/stage predictions for each requested future step, trend, slope, and latent future states. |
| Stage output | Six model classes named Benign, Reconnaissance, Initial Access, Lateral Movement, Command and Control, Exfiltration. Dataset cache labels do not necessarily provide six-class ground truth. |
| Explanation | SHAP `Explainer` applied to the dashboard's local risk-prediction function; attribution is model-specific, not causal. |
| Evaluation | Held-out and replay evaluation code, a logistic-regression baseline, and a lead-time helper; valid chronological UNSW metrics are not currently available. |
| User interface | Guided Streamlit overview explaining the model, a forecast dashboard with metric definitions, and a separate Upload & investigate workspace. |

## System workflow

```text
traffic file
  → format adapter / normalized column aliases
  → timestamp windows or Parquet row-group windows
  → 24-value traffic-state vector
  → cache normalization and latent encoding
  → causal temporal transition model
  → recursive K-step latent rollout
  → sigmoid risk score and six-logit stage output
  → dashboard / SHAP attribution / replay evaluation
```

The pipeline encodes observed history, applies the rollout-trained attack head
to that history for a displayed current score, then simulates future latent
states. The attack head evaluates prefixes of that predicted trajectory for
T+1 through T+K. The saved checkpoint has no separate current-state
supervision or calibration; all risk values are model scores, not calibrated
probabilities. Future values are model outputs, not observed traffic or ground
truth.

## Features

The order below is the exact order in `src/dataset/feature_extractor.py`.
Numeric source aliases are used when available; missing inputs commonly
default to zero. Thus a zero may mean “not present in this source,” not
“measured to be zero.” The fast Parquet cache path currently sets fragment,
port-diversity, sequential-port, and retransmission features to zero.

| Feature | Computation / source |
|---|---|
| `flow_bytes` | Sum of recognized byte/length values. |
| `flow_packets` | Sum of recognized packet-count values. |
| `flow_duration` | Maximum recognized duration in the window. |
| `iat_mean` | Mean of recognized inter-arrival-time values. |
| `iat_variance` | Population variance of those values. |
| `iat_max` | Maximum inter-arrival-time value. |
| `bidirectional_byte_ratio` | Forward/source bytes divided by backward/destination bytes; guarded for zero denominator. |
| `bidirectional_packet_ratio` | Forward/source packets divided by backward/destination packets; guarded for zero denominator. |
| `syn_count` | Sum of available SYN-flag counts. |
| `ack_count` | Sum of available ACK-flag counts. |
| `fin_count` | Sum of available FIN-flag counts. |
| `rst_count` | Sum of available RST-flag counts. |
| `psh_count` | Sum of available PSH-flag counts. |
| `urg_count` | Sum of available URG-flag counts. |
| `ttl_mean` | Mean of recognized IP TTL values. |
| `ttl_variance` | Population variance of recognized TTL values. |
| `tcp_window_mean` | Mean of recognized TCP window-size values. |
| `tcp_window_variance` | Population variance of recognized TCP window-size values. |
| `fragment_count` | Sum of source fragment indicators/counts; currently zero in fast Parquet aggregation. |
| `payload_mean` | Mean of recognized payload/packet-size values. |
| `payload_variance` | Population variance of those values. |
| `port_diversity` | Number of distinct available source/destination ports; currently zero in fast Parquet aggregation. |
| `sequential_port_score` | Fraction of adjacent sorted distinct ports differing by one; currently zero in fast Parquet aggregation. |
| `retransmission_count` | Sum of recognized retransmission counts; currently zero in fast Parquet aggregation. |

These are window-level summaries, not a complete protocol parser or a guarantee
that every dataset provides all feature fields. The PCAP adapter currently
extracts IPv4 packet length, TTL, fragment indicator, payload size, addresses,
transport ports/window, and TCP flags for TCP/UDP packets. It does not decrypt
payloads or inspect endpoint telemetry.

## Model architecture and training

The default values are defined in `src/config.py`:

- Input state: 24 features.
- Window config: 5-second window and 5-second stride for the generic
  timestamp-based extractor; sequence length 12; configured rollout horizon 5.
- Encoder: `Linear(24, 128) → GELU → Linear(128, 32)`. Decoder:
  `Linear(32, 128) → GELU → Linear(128, 24)`.
- Default transition backbone: latent input `Linear(32, 128)`, two causal
  attention blocks, four attention heads, dropout 0.1, and output
  `LayerNorm(128) → Linear(128, 32)`. Each attention block includes residual
  connections, LayerNorm, and a feed-forward `128 → 512 → 128` path with GELU
  and dropout. No positional-embedding layer is implemented.
- Optional transition backbone: two-layer LSTM with hidden size 128 and a
  `LayerNorm(128) → Linear(128, 32)` output projection.
- Attack head: `Linear(32, 128) → GELU`, max-pool over trajectory steps,
  sigmoid scalar risk head, and a six-logit stage head.
- Training defaults: Adam, learning rate 0.001, batch size 32, 20 dynamics
  epochs and 20 attack epochs. `train.py` offers epoch and batch-size overrides.

Training has two phases. Dynamics pretraining encodes the complete observed
history, predicts one latent transition, decodes it, and minimizes MSE against
the immediately following feature window. Attack fine-tuning minimizes risk
MSE plus stage cross-entropy separately for each predicted rollout prefix,
using the label from the matching future window; the encoder output supplied
to that phase is detached. The saved attack head is trained on predicted
rollouts, not separately supervised on the current observed state. Existing
checkpoints without `attack_supervision: horizon_aligned_rollout_prefixes_v1`
predate this correction and must be retrained before interpreting T+2 and
later outputs as horizon-aligned predictions. There is no validation-based
early stopping in `train.py`.
Although configuration includes a random-seed value, the training entry point
does not apply it to Python, NumPy, or PyTorch; training is not guaranteed to be
bitwise reproducible.

The `.pt` checkpoint stores encoder, world-model and attack-head weights,
normalizer state, feature names, configuration, and loss histories.

## Temporal windows, labels, and datasets

### Windowing differs by ingestion path

- Generic event streaming uses timestamp-ordered, non-overlapping windows by
  default (5 seconds/5 seconds). `stream_windows` rejects out-of-order input
  unless explicitly allowed. Batch aggregation sorts events and can make
  overlapping windows.
- The current large-file Parquet fast path ignores timestamps and groups rows
  in source order into **8,192-row windows**. It emits a final partial group.
  These are row groups, not five-second windows.
- A timestamp-less Parquet reader can assign row-index timestamps in its
  general event adapter, but compact preparation chooses the Parquet fast path.

### Formats and label behavior

| Input | Current support and caveats |
|---|---|
| CSV/TSV | Standard-library `csv.DictReader`; TSV uses tab delimiter. Generic aliases cover common flow columns. For cache labels, the reducer reads `stage`/`mitre_stage`; a plain `label` column is not normalized by the CSV adapter. |
| Parquet/PQ | PyArrow record-batch ingestion; large-file cache preparation uses the Arrow/NumPy columnar path. The fast path maps available `label`/`classlabel`/`attack_cat`/`family` fields to **binary benign/attack labels**, stored as 0/1. It does not preserve the six MITRE stages in the cache. |
| PCAP/PCAPNG | Optional Scapy `PcapReader` adapter. Requires Scapy and IPv4 packets with supported TCP/UDP fields. This path is not the Arrow fast path. |
| UNSW-NB15 | Parquet column aliases and label fields work with the generic/fast Parquet paths. The local caches include an older five-second row-order representation; see the cache-version caveat below. |
| CTU-13/Binetflow | Parquet aliases include fields such as `dur`, `tot_pkts`, `tot_bytes`, and `Family`. The local CTU cache was compacted by the fast path and its labels are binary. |
| CIC collection | Local Parquet collection was compacted by the fast path; generic CIC-style aliases are supported. The fast cache preserves binary benign/attack, not verified six-stage labels. |
| CICIOT23 | Generic CSV ingestion is present. No CICIOT23-specific adapter, locally generated cache, or validated dataset result was found; compatibility depends on its columns matching supported aliases and explicit stage labels. |

The general Parquet event adapter maps recognized attack categories through
`ATTACK_CATEGORY_TO_STAGE` and maps unknown non-benign categories to stage 1.
However, compact preparation's fast Parquet path currently produces binary
labels instead. CSV cache creation reads explicit numeric `stage` or
`mitre_stage`; its `risk` field is retained in a window object but is not the
cache label written by `build_compact_cache`.

The six output stage names are model classes. They should be called
**MITRE-aligned/MITRE-style**, not official ATT&CK ground truth, unless the
source supplies analyst-verified labels. In the currently generated fast-path
caches, training/evaluation labels are binary, so reported “stage accuracy”
against those labels is not a six-class ATT&CK accuracy.

### Large-data and cache behavior

`prepare_dataset.py` writes:

- `features.f16`: normalized `float16`, shape `(window_count, 24)`;
- `labels.i8`: one `int8` window label;
- `metadata.json`: feature names, shape, dtype, source and normalizer values;
- `dataset_report.json`: counts, file size and label/window summary.

The fast Parquet path projects recognized columns, uses Arrow batch reads
(default batch size 1,048,576 rows), then aggregates fixed row groups. Cache
construction uses repeated streaming passes. It avoids loading raw Parquet
rows as Python event objects and makes feature arrays memory-mappable.

This does **not** mean every stage is bounded-memory: CSV/PCAP dashboard upload
inference aggregates records in memory; `evaluate.py` stacks sequences; and
replay currently converts the selected cache to a full `float32` array. Current
local processed caches fit the demonstrated environment, but multi-gigabyte
end-to-end model evaluation has not been established.

**Cache reproducibility caveat:** existing workspace artifacts were built by
different generations of the preprocessing code. For example,
`data/cache/unsw_nb15` has 51,535 windows, while current fast Parquet
preparation would group that source in 8,192-row blocks. Metadata does not
record the windowing algorithm/version. Do not assume an old cache, checkpoint,
and newly rebuilt cache are interchangeable. Preserve the exact cache and
checkpoint used for any reproduced report; the replay JSON includes SHA-256
hashes for them.

## Forecasting and evaluation

`src/forecasting.py` contains:

- `forecast_from_models`: observed history → current risk/stage → recursive
  future latent states → per-prefix risk/stage trajectory.
- `replay_forecasts`: deterministic origins; at horizon `k`, compares the
  forecast with the label at `origin + k`.
- `k_horizon_metrics`: precision, recall, F1, FPR, ROC-AUC, PR-AUC and
  confusion counts; undefined metrics are represented as `null`.
- `forecast_lead_time`: treats each contiguous positive label run as an event,
  finds its onset, then measures onset index minus the most recent threshold
  alert since the prior event ended.

Lead time is valid only relative to sequential, chronologically ordered labels
and the chosen alert stream. Its code accepts a `step_seconds` multiplier, but
row-order positions do not justify conversion to wall-clock time.

### Current UNSW validation status

**No genuine chronological forecast metrics or lead-time result are currently
validated for the local UNSW-NB15 data.** The source Parquet files have no
timestamps, and the existing cache combines rows in file/source order rather
than verified temporal windows. The replay artifact is retained as a
row-order label comparison only; its `data_assessment` explicitly says that
this is not validated temporal forecasting. Binary cache labels also cannot
validate six-class MITRE-stage predictions.

Previously documented held-out/replay scores and numerical lead-time values
are withdrawn: the held-out evaluation used inconsistent normalization, the
old replay did not establish chronology, compared incompatible binary labels
to stage outputs, and used a checkpoint without horizon-specific targets; the
lead-time calculation used synthetic row positions. The lead-time artifact
now marks the result unavailable. The
current evaluator applies the checkpoint's training normalizer consistently,
but the current fast grouping produces too few test windows for this
checkpoint's 12-window history plus next-window target, so it exits with a
clear insufficient-window error instead of emitting metrics.

The replay JSON records data-assessment metadata and hashes the exact cache
and checkpoint. The current local checkpoint also predates horizon-aligned
attack supervision; its T+2 and later replay metrics are exploratory even as
row-order label comparisons. A valid early-warning claim requires a newly
trained horizon-aligned checkpoint, timestamped and ordered traffic windows,
sequential labels with defined attack onset, and forecasts aligned to actual
issuance times.

## Setup and commands (Windows PowerShell)

Tested in the current workspace on Windows with Python 3.14.5 and the
project-local `.venv`. `requirements.txt` specifies minimum versions rather
than a fully locked environment. No database or environment variables are
required. No license or citation file is present in this repository.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local project virtual environment successfully imports PyTorch in this
workspace. A separate global/user-site PyTorch installation was previously
blocked by Windows Application Control (`WinError 4551` loading
`torch_global_deps.dll`). If model import fails, resolve the local PyTorch
installation/policy; preparation and cache inspection do not need model
inference. The repository does not implement a fake-model fallback.

### Prepare a dataset

Replace `<input-file-or-directory>` with your local source path:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py `
  --input "<input-file-or-directory>" `
  --output "data\cache\my_dataset"
```

`--input` accepts supported files or a directory; `--output` is the cache and
report destination. Example for a local Parquet directory:

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py `
  --input "<path-to-UNSW-parquet-directory>" `
  --output "data\cache\unsw_nb15"
```

### Train a checkpoint

```powershell
.\.venv\Scripts\python.exe train.py `
  --data "<path-to-labeled-training-file>" `
  --output "artifacts\unsw_nb15_quick" `
  --dynamics-epochs 1 `
  --attack-epochs 1 `
  --batch-size 256
```

The one-epoch overrides are a smoke/demo command, not a recommended final
training recipe. The checkpoint is written as `world_model.pt`; training also
builds a cache under the output directory.

### Run a forecast demo

```powershell
.\.venv\Scripts\python.exe demo_forecast.py `
  --checkpoint "artifacts\unsw_nb15_quick\world_model.pt" `
  --cache "data\cache\unsw_nb15" `
  --horizon 3
```

### Replay against subsequent labeled windows

```powershell
.\.venv\Scripts\python.exe replay_forecast.py `
  --checkpoint "artifacts\unsw_nb15_quick\world_model.pt" `
  --cache "data\cache\unsw_nb15" `
  --output "artifacts\unsw_nb15_quick\forecast_replay.json" `
  --horizon 3 `
  --threshold 0.5 `
  --max-origins 1000
```

`--max-origins 0` evaluates all possible origins and may take substantially
longer. The report is JSON. `step-seconds` is configurable, but must not be used
to imply wall time for synthetic row order.

### Held-out evaluation

```powershell
.\.venv\Scripts\python.exe evaluate.py `
  --checkpoint "artifacts\unsw_nb15_quick\world_model.pt" `
  --train-data "<path-to-UNSW_NB15_training-set.parquet>" `
  --test-data "<path-to-UNSW_NB15_testing-set.parquet>" `
  --output "artifacts\unsw_nb15_quick\evaluation"
```

### Start the dashboard and tests

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.headless true --server.port 8501
.\.venv\Scripts\python.exe -m pytest -q
```

Open `http://localhost:8501`. The **Overview** explains the model and its
limits. **Forecast dashboard** reads an existing local cache; **Upload &
investigate** accepts a file for optional inference. Metric tooltips and the
Metric guide explain the figures and their limitations. The upload path
aggregates the supplied input in memory and is not the large-file preparation
path.

## Project structure

```text
app.py                         Streamlit live forecast and upload UI
prepare_dataset.py             Dataset profiling and compact-cache CLI
train.py                       Two-phase checkpoint training
evaluate.py                    Held-out model / logistic evaluation
demo_forecast.py               Single forecast from cache/checkpoint
replay_forecast.py             Sequential forecast-vs-label replay report
src/config.py                  Feature/model/window/training defaults and stage map
src/dataset/ingestion.py       CSV, TSV, Parquet, PCAP adapters
src/dataset/feature_extractor.py 24 features, windows, normalization
src/dataset/large_files.py     Parquet fast path, profile, compact cache
src/models/encoder.py          State autoencoder
src/models/world_model.py      Transformer/LSTM transition and rollout
src/models/attack_head.py      Risk and stage heads
src/forecasting.py             Forecast summary, replay, metrics, lead time
src/explainability/explainer.py SHAP and text-summary helpers
src/benchmarks/baseline.py     Logistic regression and metric helpers
tests/                         Feature-pipeline and forecasting tests
docs/FORECASTING.md            Forecast/replay API details
requirements.txt               Python dependency minimums
```

## Limitations and interpretation

- A forecast is not proof that an attack occurred or will occur.
- Sigmoid risk is a model score; calibration as a real-world probability has
  not been established.
- No attacker identity, endpoint process/file/user/credential telemetry, or
  encrypted-payload decryption is provided.
- No live network-interface capture or continuous online monitoring is
  implemented; input is a local file or prepared cache.
- The dashboard assists analysis; it does not replace a SOC analyst or respond
  to incidents.
- Several labels are binary or heuristically mapped; six-class ATT&CK
  validation is not established.
- Timestamp-less Parquet uses row-group order. Lead time is then measured in
  windows, not wall-clock time.
- The existing replay result is sampled from the same cache and is not an
  independent held-out evaluation.
- The current cache metadata does not version the aggregation procedure;
  legacy and current fast-path caches can have different temporal semantics.
- SHAP attribution describes model sensitivity/attribution, not causality.
- Several features may be zero because a source does not provide them.
- CICIOT23-specific schema support and PCAP/PCAPNG datasets were not validated
  in the local generated result set.
- Training has no implemented validation/early-stopping loop or applied random
  seed; cross-dataset generalization and production suitability are unproven.

For a more detailed forecast/replay explanation, see
[docs/FORECASTING.md](docs/FORECASTING.md).
