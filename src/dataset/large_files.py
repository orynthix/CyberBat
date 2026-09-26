"""Bounded-memory profiling and compact caching for large traffic datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .feature_extractor import FEATURE_NAMES, FeatureNormalizer, TrafficFeatureExtractor, WindowFeatures
from .ingestion import load_events

FAST_PARQUET_ROWS_PER_WINDOW = 8_192


def profile_dataset(
    path: str | Path,
    output_path: str | Path | None = None,
    sample_limit: int = 10_000,
) -> dict[str, Any]:
    """Scan a dataset once while retaining only counters and bounded samples."""

    if sample_limit < 0:
        raise ValueError("sample_limit cannot be negative")
    extractor = TrafficFeatureExtractor()
    event_count = 0
    window_count = 0
    malformed_labels = 0
    stage_counts: Counter[str] = Counter()
    risk_min = 1.0
    risk_max = 0.0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    windows_iter = _fast_parquet_windows(path) if _is_parquet_source(path) else extractor.stream_windows(load_events(path), sample_size=0)
    for window in windows_iter:
        window_count += 1
        event_count += window.event_count
        first_timestamp = window.start if first_timestamp is None else min(first_timestamp, window.start)
        last_timestamp = window.end if last_timestamp is None else max(last_timestamp, window.end)
        stage_counts[str(window.stage)] += window.event_count
        if window.risk:
            risk_min = min(risk_min, window.risk)
            risk_max = max(risk_max, window.risk)
        else:
            risk_min = 0.0
    report: dict[str, Any] = {
        "path": str(Path(path).resolve()),
        "file_size_bytes": (
            Path(path).stat().st_size
            if Path(path).is_file()
            else sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())
        ),
        "event_count": event_count,
        "window_count": window_count,
        "feature_count": len(FEATURE_NAMES),
        "first_timestamp": first_timestamp,
        "last_timestamp": last_timestamp,
        "stage_counts": dict(stage_counts),
        "risk_min": risk_min if event_count else None,
        "risk_max": risk_max if event_count else None,
        "malformed_labels": malformed_labels,
        "memory_strategy": "streaming windows with bounded samples; raw rows are not retained",
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def build_compact_cache(
    path: str | Path,
    output_dir: str | Path,
    window_seconds: float = 5.0,
    stride_seconds: float = 5.0,
    normalizer: FeatureNormalizer | None = None,
) -> dict[str, Any]:
    """Create a compact memory-mapped feature/label cache using two streaming passes."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    extractor = TrafficFeatureExtractor(window_seconds, stride_seconds)
    parquet_fast_path = _is_parquet_source(path)

    def windows() -> Any:
        if parquet_fast_path:
            return _fast_parquet_windows(path)
        return extractor.stream_windows(load_events(path), sample_size=0)

    window_count = sum(1 for _ in windows())
    if window_count == 0:
        raise ValueError("input contains no usable traffic windows")
    feature_path = destination / "features.f16"
    label_path = destination / "labels.i8"
    normalizer = normalizer or FeatureNormalizer()
    if not normalizer.fitted:
        total = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        squares = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        for window in windows():
            chunk = np.asarray(window.values, dtype=np.float64)
            total += chunk.sum(axis=0)
            squares += np.square(chunk).sum(axis=0)
        normalizer.mean_ = (total / window_count).astype(np.float32)
        normalizer.scale_ = np.sqrt(np.maximum(squares / window_count - np.square(total / window_count), 0.0)).astype(np.float32)
        normalizer.scale_[normalizer.scale_ < 1e-8] = 1.0
    features = np.memmap(feature_path, dtype="float16", mode="w+", shape=(window_count, len(FEATURE_NAMES)))
    labels = np.memmap(label_path, dtype="int8", mode="w+", shape=(window_count,))
    index = 0
    for window in windows():
        features[index] = normalizer.transform(window.values[None, :])[0].astype(np.float16)
        labels[index] = window.stage
        index += 1
    features.flush()
    labels.flush()
    metadata = {
        "source": str(Path(path).resolve()),
        "window_count": window_count,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "features_file": feature_path.name,
        "labels_file": label_path.name,
        "features_dtype": "float16",
        "features_normalized": True,
        "normalizer": normalizer.state_dict(),
        "label_semantics": "binary_attack" if parquet_fast_path else "source_stage",
        "windowing": (
            {
                "method": "fixed_source_order_row_groups",
                "timestamp_used": False,
                "rows_per_window": FAST_PARQUET_ROWS_PER_WINDOW,
            }
            if parquet_fast_path
            else {
                "method": "streamed_event_timestamps",
                "timestamp_used": None,
                "timestamp_order_verified": False,
                "window_seconds": window_seconds,
                "stride_seconds": stride_seconds,
            }
        ),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _is_parquet_source(path: str | Path) -> bool:
    source = Path(path)
    return source.is_file() and source.suffix.lower() in {".parquet", ".pq"} or (
        source.is_dir() and any(file.suffix.lower() in {".parquet", ".pq"} for file in source.rglob("*"))
    )


def _fast_parquet_windows(
    path: str | Path,
    batch_size: int = 1_048_576,
    rows_per_window: int = FAST_PARQUET_ROWS_PER_WINDOW,
) -> Any:
    """Yield compact windows using Arrow columns, avoiding Python row objects.

    When a Parquet file has no timestamp, rows are grouped into deterministic
    windows. This prevents flow datasets from creating one temporal window per
    row while preserving stable source order.
    """

    try:
        import pyarrow as pa
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("fast Parquet processing requires pyarrow") from exc
    if rows_per_window < 1:
        raise ValueError("rows_per_window must be positive")
    sources = [Path(path)] if Path(path).is_file() else sorted(
        file for file in Path(path).rglob("*") if file.suffix.lower() in {".parquet", ".pq"}
    )
    global_index = 0
    for source in sources:
        parquet_file = parquet.ParquetFile(source)
        columns = parquet_file.schema.names
        normalized = {name.strip().lower().replace(" ", "_").replace("-", "_"): name for name in columns}
        wanted = {
            "bytes", "flow_bytes", "total_bytes", "tot_bytes", "totlen_fwd_pkts",
            "fwd_packets_length_total", "packets", "flow_packets", "total_packets",
            "tot_pkts", "total_fwd_packets", "flow_duration", "duration", "dur",
            "iat", "inter_arrival_time", "flow_iat_mean", "ttl", "ip_ttl", "ttl_mean",
            "tcp_window", "window_size", "tcp_window_size", "init_fwd_win_bytes",
            "payload_size", "payload_bytes", "avg_packet_size", "packet_length_mean",
            "src_bytes", "forward_bytes", "sbytes", "totlen_bwd_pkts",
            "bwd_packets_length_total", "dst_bytes", "backward_bytes", "dbytes",
            "src_packets", "forward_packets", "spkts", "dst_packets", "backward_packets",
            "dpkts", "total_backward_packets", "syn_count", "tcp_syn_count",
            "syn_flag_count", "ack_count", "tcp_ack_count", "ack_flag_count",
            "fin_count", "tcp_fin_count", "fin_flag_count", "rst_count",
            "tcp_rst_count", "rst_flag_count", "psh_count", "tcp_psh_count",
            "fwd_psh_flags", "urg_count", "tcp_urg_count", "urg_flag_count",
            "label", "classlabel", "attack_cat", "family",
        }
        selected_columns = [actual for key, actual in normalized.items() if key in wanted]
        selected_index = {name: index for index, name in enumerate(selected_columns)}
        pending: dict[str, list[np.ndarray]] = {}
        pending_labels: list[np.ndarray] = []
        pending_count = 0
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=selected_columns,
            use_threads=True,
        ):
            table = batch
            row_count = batch.num_rows

            def numeric(*names: str) -> np.ndarray:
                for name in names:
                    actual = normalized.get(name)
                    if actual is not None:
                        values = table.column(selected_index[actual]).to_numpy(zero_copy_only=False)
                        return np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
                return np.zeros(row_count, dtype=np.float64)

            def risk_array(*names: str) -> np.ndarray:
                import pyarrow.compute as pc

                for name in names:
                    actual = normalized.get(name)
                    if actual is not None:
                        column = pc.cast(table.column(selected_index[actual]), "string")
                        benign = pc.is_in(
                            pc.utf8_lower(pc.fill_null(column, "")),
                            value_set=pa.array(["0", "0.0", "normal", "benign", "background"]),
                        )
                        return np.logical_not(benign.to_numpy(zero_copy_only=False)).astype(np.float64)
                return np.zeros(row_count, dtype=np.float64)

            arrays = {
                "bytes": numeric("bytes", "flow_bytes", "total_bytes", "tot_bytes", "totlen_fwd_pkts", "fwd_packets_length_total"),
                "packets": numeric("packets", "flow_packets", "total_packets", "tot_pkts", "total_fwd_packets"),
                "duration": numeric("flow_duration", "duration", "dur"),
                "iat": numeric("iat", "inter_arrival_time", "flow_iat_mean"),
                "ttl": numeric("ttl", "ip_ttl", "ttl_mean"),
                "tcp_window": numeric("tcp_window", "window_size", "tcp_window_size", "init_fwd_win_bytes"),
                "payload": numeric("payload_size", "payload_bytes", "avg_packet_size", "packet_length_mean"),
                "src_bytes": numeric("src_bytes", "forward_bytes", "sbytes", "totlen_fwd_pkts", "fwd_packets_length_total"),
                "dst_bytes": numeric("dst_bytes", "backward_bytes", "dbytes", "totlen_bwd_pkts", "bwd_packets_length_total"),
                "src_packets": numeric("src_packets", "forward_packets", "spkts", "total_fwd_packets"),
                "dst_packets": numeric("dst_packets", "backward_packets", "dpkts", "total_backward_packets"),
                "syn": numeric("syn_count", "tcp_syn_count", "syn_flag_count"),
                "ack": numeric("ack_count", "tcp_ack_count", "ack_flag_count"),
                "fin": numeric("fin_count", "tcp_fin_count", "fin_flag_count"),
                "rst": numeric("rst_count", "tcp_rst_count", "rst_flag_count"),
                "psh": numeric("psh_count", "tcp_psh_count", "fwd_psh_flags"),
                "urg": numeric("urg_count", "tcp_urg_count", "urg_flag_count"),
                "risk": np.zeros(row_count, dtype=np.float64),
            }
            arrays["risk"] = risk_array("label", "classlabel", "attack_cat", "family")
            pending_count += row_count
            for key, values in arrays.items():
                pending.setdefault(key, []).append(values)
            pending_labels.append((arrays["risk"] > 0).astype(np.int8))
            while pending_count >= rows_per_window:
                merged = {key: np.concatenate(values) for key, values in pending.items()}
                take = rows_per_window
                yield _vector_window(merged, pending_labels, take, global_index)
                global_index += take
                pending = {key: [values[take:]] for key, values in merged.items()}
                pending_labels = [np.concatenate(pending_labels)[take:]]
                pending_count -= take
        if pending_count:
            merged = {key: np.concatenate(values) for key, values in pending.items()}
            yield _vector_window(merged, pending_labels, pending_count, global_index)
            global_index += pending_count


def _vector_window(arrays: dict[str, np.ndarray], labels: list[np.ndarray], count: int, start: int) -> WindowFeatures:
    values = np.asarray(
        [
            arrays["bytes"].sum(),
            arrays["packets"].sum(),
            arrays["duration"].max(initial=0.0),
            arrays["iat"].mean(),
            arrays["iat"].var(),
            arrays["iat"].max(initial=0.0),
            arrays["src_bytes"].sum() / max(arrays["dst_bytes"].sum(), 1.0),
            arrays["src_packets"].sum() / max(arrays["dst_packets"].sum(), 1.0),
            arrays["syn"].sum(), arrays["ack"].sum(), arrays["fin"].sum(), arrays["rst"].sum(),
            arrays["psh"].sum(), arrays["urg"].sum(), arrays["ttl"].mean(), arrays["ttl"].var(),
            arrays["tcp_window"].mean(), arrays["tcp_window"].var(), 0.0,
            arrays["payload"].mean(), arrays["payload"].var(), 0.0, 0.0, 0.0,
        ],
        dtype=np.float32,
    )
    label_array = np.concatenate(labels)[:count]
    stage = int(label_array.max(initial=0))
    return WindowFeatures(float(start), float(start + count), values, (), count, stage, float(stage > 0))
