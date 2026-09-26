import numpy as np

from src.dataset.feature_extractor import (
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    FeatureNormalizer,
    TrafficFeatureExtractor,
)
from src.dataset.large_files import _fast_parquet_windows, build_compact_cache, profile_dataset
from src.dataset.ingestion import load_events


def test_missing_packet_attributes_are_safe_and_shape_is_stable() -> None:
    events = [
        {"timestamp": 0.0, "bytes": 100, "packets": 2, "syn_count": 1},
        {"timestamp": 1.0, "bytes": 50, "packets": 1, "dst_port": 80},
    ]
    matrix = TrafficFeatureExtractor(window_seconds=5).aggregate(events)

    assert matrix.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(matrix).all()
    assert matrix[0, 0] == 150


def test_every_model_feature_has_a_dashboard_description() -> None:
    assert set(FEATURE_DESCRIPTIONS) == set(FEATURE_NAMES)
    assert all(FEATURE_DESCRIPTIONS[name] for name in FEATURE_NAMES)


def test_sequences_are_normalized_and_have_expected_shape() -> None:
    events = [
        {"timestamp": float(index), "bytes": index + 1, "packets": 1, "ttl": 64}
        for index in range(6)
    ]
    extractor = TrafficFeatureExtractor(window_seconds=1, stride_seconds=1)
    matrix = extractor.aggregate(events)
    sequences = extractor.to_sequences(matrix, sequence_length=3)

    assert sequences.shape == (4, 3, len(FEATURE_NAMES))
    assert np.isfinite(sequences).all()
    ttl_index = FEATURE_NAMES.index("ttl_mean")
    assert np.allclose(sequences[:, :, ttl_index], 0.0, atol=1e-6)
    assert extractor.normalizer.fitted


def test_streaming_windows_do_not_retain_all_events(tmp_path) -> None:
    rows = [
        {"timestamp": float(index), "bytes": 10, "packets": 1, "stage": 1 if index > 4 else 0}
        for index in range(10)
    ]
    extractor = TrafficFeatureExtractor(window_seconds=5, stride_seconds=5)
    windows = list(extractor.stream_windows(rows, sample_size=1))

    assert len(windows) == 2
    assert sum(window.event_count for window in windows) == 10
    assert all(len(window.events) <= 1 for window in windows)


def test_compact_cache_contains_only_features_labels_and_metadata(tmp_path) -> None:
    source = tmp_path / "traffic.csv"
    source.write_text(
        "timestamp,bytes,packets,stage\n"
        + "".join(f"{index * 5},{index + 1},1,{int(index > 1)}\n" for index in range(4)),
        encoding="utf-8",
    )
    report = profile_dataset(source)
    metadata = build_compact_cache(source, tmp_path / "cache")

    assert report["event_count"] == 4
    assert metadata["window_count"] == 4
    assert (tmp_path / "cache" / "features.f16").stat().st_size == 4 * len(FEATURE_NAMES) * 2
    assert metadata["windowing"]["method"] == "streamed_event_timestamps"
    assert metadata["windowing"]["timestamp_order_verified"] is False


def test_compact_cache_can_reuse_training_normalizer(tmp_path) -> None:
    source = tmp_path / "traffic.csv"
    source.write_text(
        "timestamp,bytes,packets,stage\n"
        "0,10,1,0\n"
        "5,20,1,1\n",
        encoding="utf-8",
    )
    mean = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    scale = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    mean[0] = 10.0
    scale[0] = 10.0
    normalizer = FeatureNormalizer()
    normalizer.mean_ = mean
    normalizer.scale_ = scale

    metadata = build_compact_cache(source, tmp_path / "cache", normalizer=normalizer)
    features = np.memmap(
        tmp_path / "cache" / metadata["features_file"],
        dtype="float16",
        mode="r",
        shape=(2, len(FEATURE_NAMES)),
    )

    assert metadata["normalizer"] == normalizer.state_dict()
    assert np.allclose(features[:, 0], [0.0, 1.0])


def test_parquet_batches_are_streamed_and_labels_normalized(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "traffic.parquet"
    table = pa.table(
        {
            "dur": [0.1, 0.2, 0.3],
            "tot_pkts": [2, 3, 4],
            "tot_bytes": [20, 30, 40],
            "label": ["Benign", "Attack", "Attack"],
            "Family": ["", "Botnet", "Botnet"],
        }
    )
    pq.write_table(table, source)
    events = list(load_events(source))

    assert [event.timestamp for event in events] == [0.0, 1.0, 2.0]
    assert [event.values["stage"] for event in events] == [0, 4, 4]
    assert events[1].values["risk"] == 1.0


def test_arrow_fast_path_aggregates_rows_without_materializing_events(tmp_path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "fast.parquet"
    pq.write_table(
        pa.table(
            {
                "dur": [0.1, 0.2, 0.3],
                "tot_pkts": [2, 3, 4],
                "tot_bytes": [20, 30, 40],
                "label": ["Benign", "Attack", "Attack"],
            }
        ),
        source,
    )

    windows = list(_fast_parquet_windows(source, rows_per_window=2))

    assert len(windows) == 2
    assert [window.event_count for window in windows] == [2, 1]
    assert windows[0].values.shape == (len(FEATURE_NAMES),)
    assert windows[0].stage == 1
    metadata = build_compact_cache(source, tmp_path / "cache")
    assert metadata["label_semantics"] == "binary_attack"
    assert metadata["windowing"]["timestamp_used"] is False
    assert metadata["windowing"]["rows_per_window"] == 8192
