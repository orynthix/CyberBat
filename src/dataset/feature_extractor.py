"""Dual-level traffic feature extraction and temporal window aggregation.

The extractor accepts dictionaries or dataframe-like row mappings so ingestion
adapters can remain independent of packet parsing libraries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "flow_bytes",
    "flow_packets",
    "flow_duration",
    "iat_mean",
    "iat_variance",
    "iat_max",
    "bidirectional_byte_ratio",
    "bidirectional_packet_ratio",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "psh_count",
    "urg_count",
    "ttl_mean",
    "ttl_variance",
    "tcp_window_mean",
    "tcp_window_variance",
    "fragment_count",
    "payload_mean",
    "payload_variance",
    "port_diversity",
    "sequential_port_score",
    "retransmission_count",
)

FEATURE_DESCRIPTIONS = {
    "flow_bytes": "Total recognized bytes/length summed in the window.",
    "flow_packets": "Total recognized packet counts summed in the window.",
    "flow_duration": "Maximum recognized flow duration in the window.",
    "iat_mean": "Mean of available inter-arrival-time values.",
    "iat_variance": "Population variance of available inter-arrival-time values.",
    "iat_max": "Largest available inter-arrival-time value.",
    "bidirectional_byte_ratio": "Forward/source bytes divided by backward/destination bytes; guarded for zero denominator.",
    "bidirectional_packet_ratio": "Forward/source packets divided by backward/destination packets; guarded for zero denominator.",
    "syn_count": "Sum of available TCP SYN counts.",
    "ack_count": "Sum of available TCP ACK counts.",
    "fin_count": "Sum of available TCP FIN counts.",
    "rst_count": "Sum of available TCP RST counts.",
    "psh_count": "Sum of available TCP PSH counts.",
    "urg_count": "Sum of available TCP URG counts.",
    "ttl_mean": "Mean of recognized IP time-to-live values.",
    "ttl_variance": "Population variance of recognized IP time-to-live values.",
    "tcp_window_mean": "Mean of recognized TCP receive-window sizes.",
    "tcp_window_variance": "Population variance of recognized TCP receive-window sizes.",
    "fragment_count": "Sum of recognized IP fragment indicators/counts.",
    "payload_mean": "Mean of recognized payload or packet-size values.",
    "payload_variance": "Population variance of recognized payload or packet-size values.",
    "port_diversity": "Count of distinct source/destination ports available in the window.",
    "sequential_port_score": "Fraction of adjacent sorted unique ports that differ by one.",
    "retransmission_count": "Sum of recognized retransmission counts.",
}


def _normalized_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key).strip().lower().replace(" ", "_").replace("-", "_"): value
        for key, value in row.items()
    }


def _number(row: Mapping[str, Any], *names: str, default: float = 0.0) -> float:
    """Read the first numeric value present, returning a safe default."""

    normalized = row if all(name in row for name in names) else _normalized_mapping(row)
    for name in names:
        value = row.get(name, normalized.get(name.strip().lower().replace(" ", "_").replace("-", "_")))
        if value is not None and value != "":
            try:
                result = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(result):
                return result
    return default


def _values(row: Mapping[str, Any], *names: str) -> list[float]:
    value: Any = None
    normalized = _normalized_mapping(row)
    for name in names:
        candidate = row.get(name, normalized.get(name.strip().lower().replace(" ", "_").replace("-", "_")))
        if candidate is not None:
            value = candidate
            break
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        raw_values = value
    else:
        raw_values = str(value).replace(";", ",").split(",")
    result: list[float] = []
    for item in raw_values:
        try:
            parsed = float(item)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed):
            result.append(parsed)
    return result


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    return float(np.var(values)) if values else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


@dataclass(frozen=True)
class TrafficEvent:
    """Canonical event used by the extractor and future ingestion adapters."""

    timestamp: float
    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "TrafficEvent":
        timestamp = _number(row, "timestamp", "ts", "time", default=0.0)
        return cls(timestamp=timestamp, values=row)


@dataclass(frozen=True)
class WindowFeatures:
    """Features and source events belonging to one time window."""

    start: float
    end: float
    values: np.ndarray
    events: tuple[TrafficEvent, ...]
    event_count: int = 0
    stage: int = 0
    risk: float = 0.0


class _OnlineStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.square_total = 0.0
        self.maximum = 0.0

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        self.total += value
        self.square_total += value * value
        self.maximum = max(self.maximum, value)

    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def variance(self) -> float:
        if not self.count:
            return 0.0
        return max(0.0, self.square_total / self.count - self.mean() ** 2)


class _WindowAccumulator:
    """Bounded-memory online reducer for one temporal window."""

    def __init__(self, start: float, end: float, sample_size: int) -> None:
        self.start = start
        self.end = end
        self.count = 0
        self.bytes = 0.0
        self.packets = 0.0
        self.duration = 0.0
        self.source_bytes = 0.0
        self.destination_bytes = 0.0
        self.source_packets = 0.0
        self.destination_packets = 0.0
        self.flags = {name: 0.0 for name in ("syn_count", "ack_count", "fin_count", "rst_count", "psh_count", "urg_count")}
        self.fragments = 0.0
        self.retransmissions = 0.0
        self.iats = _OnlineStats()
        self.ttl = _OnlineStats()
        self.tcp_windows = _OnlineStats()
        self.payloads = _OnlineStats()
        self.ports: set[int] = set()
        self.sample: deque[TrafficEvent] = deque(maxlen=sample_size)
        self.stage = 0
        self.risk = 0.0

    def add(self, event: TrafficEvent) -> None:
        row = _normalized_mapping(event.values)
        self.count += 1
        self.sample.append(event)
        try:
            self.stage = max(self.stage, int(float(row.get("stage", row.get("mitre_stage", 0)) or 0)))
        except (TypeError, ValueError):
            pass
        try:
            self.risk = max(self.risk, float(row.get("risk", row.get("infiltration_risk", 0.0)) or 0.0))
        except (TypeError, ValueError):
            pass
        self.bytes += _number(row, "bytes", "flow_bytes", "total_bytes", "tot_bytes", "totlen_fwd_pkts", "total_length_of_fwd_packets")
        self.packets += _number(row, "packets", "flow_packets", "total_packets", "tot_pkts", "tot_fwd_pkts", "total_fwd_packets")
        self.duration = max(self.duration, _number(row, "flow_duration", "duration", "dur", "flow_duration_us"))
        self.source_bytes += _number(row, "src_bytes", "forward_bytes", "sbytes", "totlen_fwd_pkts")
        self.destination_bytes += _number(row, "dst_bytes", "backward_bytes", "dbytes", "totlen_bwd_pkts")
        self.source_packets += _number(row, "src_packets", "forward_packets", "spkts", "tot_fwd_pkts")
        self.destination_packets += _number(row, "dst_packets", "backward_packets", "dpkts", "tot_bwd_pkts")
        for name, aliases in {
            "syn_count": ("syn_count", "tcp_syn_count", "syn_flag_count"),
            "ack_count": ("ack_count", "tcp_ack_count", "ack_flag_count"),
            "fin_count": ("fin_count", "tcp_fin_count", "fin_flag_count"),
            "rst_count": ("rst_count", "tcp_rst_count", "rst_flag_count"),
            "psh_count": ("psh_count", "tcp_psh_count", "fwd_psh_flags"),
            "urg_count": ("urg_count", "tcp_urg_count", "urg_flag_count"),
        }.items():
            self.flags[name] += _number(row, *aliases)
        self.fragments += _number(row, "fragment_count", "ip_fragments")
        self.retransmissions += _number(row, "retransmissions", "retransmission_count")
        for stats, names in (
            (self.iats, ("iat", "inter_arrival_time", "flow_iat_mean")),
            (self.ttl, ("ttl", "ip_ttl", "ttl_mean")),
            (self.tcp_windows, ("tcp_window", "window_size", "tcp_window_size")),
            (self.payloads, ("payload_size", "payload_bytes", "avg_packet_size")),
        ):
            values = _values(row, *names)
            stats.add(values[0] if values else _number(row, *names))
        for name in ("dst_port", "destination_port", "dst_port_number", "src_port", "source_port"):
            for value in _values(row, name):
                if value > 0 and len(self.ports) < 100_000:
                    self.ports.add(int(value))

    def finish(self) -> WindowFeatures:
        ordered_ports = sorted(self.ports)
        sequential = (
            sum(b - a == 1 for a, b in zip(ordered_ports, ordered_ports[1:])) / (len(ordered_ports) - 1)
            if len(ordered_ports) > 1 else 0.0
        )
        values = np.asarray(
            [
                self.bytes,
                self.packets,
                self.duration,
                self.iats.mean(),
                self.iats.variance(),
                self.iats.maximum,
                _ratio(self.source_bytes, self.destination_bytes),
                _ratio(self.source_packets, self.destination_packets),
                self.flags["syn_count"],
                self.flags["ack_count"],
                self.flags["fin_count"],
                self.flags["rst_count"],
                self.flags["psh_count"],
                self.flags["urg_count"],
                self.ttl.mean(),
                self.ttl.variance(),
                self.tcp_windows.mean(),
                self.tcp_windows.variance(),
                self.fragments,
                self.payloads.mean(),
                self.payloads.variance(),
                float(len(self.ports)),
                sequential,
                self.retransmissions,
            ],
            dtype=np.float32,
        )
        return WindowFeatures(self.start, self.end, values, tuple(self.sample), self.count, self.stage, self.risk)


class FeatureNormalizer:
    """Training-only statistics used to standardize window feature vectors."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    @property
    def fitted(self) -> bool:
        return self.mean_ is not None and self.scale_ is not None

    def fit(self, matrix: np.ndarray) -> "FeatureNormalizer":
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected a 2-D matrix with {len(FEATURE_NAMES)} features")
        if values.shape[0] == 0:
            raise ValueError("cannot fit a normalizer on an empty matrix")
        self.mean_ = np.nan_to_num(values.mean(axis=0), nan=0.0)
        scale = np.nan_to_num(values.std(axis=0), nan=0.0)
        self.scale_ = np.where(scale < 1e-8, 1.0, scale).astype(np.float32)
        return self

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before transform")
        values = np.asarray(matrix, dtype=np.float32)
        assert self.mean_ is not None and self.scale_ is not None
        return ((values - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, matrix: np.ndarray) -> np.ndarray:
        """Return feature values to the original scale."""

        if not self.fitted:
            raise RuntimeError("FeatureNormalizer must be fitted before inverse_transform")
        values = np.asarray(matrix, dtype=np.float32)
        assert self.mean_ is not None and self.scale_ is not None
        return (values * self.scale_ + self.mean_).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted normalizer")
        assert self.mean_ is not None and self.scale_ is not None
        return {"mean": self.mean_.tolist(), "scale": self.scale_.tolist()}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "FeatureNormalizer":
        normalizer = cls()
        normalizer.mean_ = np.asarray(state["mean"], dtype=np.float32)
        normalizer.scale_ = np.asarray(state["scale"], dtype=np.float32)
        if normalizer.mean_.shape != (len(FEATURE_NAMES),) or normalizer.scale_.shape != (len(FEATURE_NAMES),):
            raise ValueError("normalizer state has an invalid feature dimension")
        return normalizer

    def fit_transform(self, matrix: np.ndarray) -> np.ndarray:
        return self.fit(matrix).transform(matrix)


class TrafficFeatureExtractor:
    """Aggregate canonical traffic events into normalized temporal sequences."""

    def __init__(
        self,
        window_seconds: float = 5.0,
        stride_seconds: float = 5.0,
        normalizer: FeatureNormalizer | None = None,
    ) -> None:
        if window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.normalizer = normalizer or FeatureNormalizer()

    @staticmethod
    def _aggregate(events: Sequence[TrafficEvent]) -> np.ndarray:
        rows = [event.values for event in events]
        bytes_values = [_number(row, "bytes", "flow_bytes", "total_bytes", "tot_bytes", "totlen_fwd_pkts", "total_length_of_fwd_packets") for row in rows]
        packet_values = [_number(row, "packets", "flow_packets", "total_packets", "tot_pkts", "tot_fwd_pkts", "total_fwd_packets") for row in rows]
        iats = [value for row in rows for value in (_values(row, "iat", "inter_arrival_time", "flow_iat_mean") or [_number(row, "iat", "inter_arrival_time", "flow_iat_mean")]) if value > 0]
        ttl = [value for row in rows for value in (_values(row, "ttl", "ip_ttl", "ttl_mean") or [_number(row, "ttl", "ip_ttl", "ttl_mean")]) if value > 0]
        tcp_windows = [value for row in rows for value in (_values(row, "tcp_window", "window_size", "tcp_window_size") or [_number(row, "tcp_window", "window_size", "tcp_window_size")]) if value > 0]
        payloads = [value for row in rows for value in (_values(row, "payload_size", "payload_bytes", "avg_packet_size") or [_number(row, "payload_size", "payload_bytes", "avg_packet_size")])]
        ports = [value for row in rows for value in (_values(row, "dst_port", "destination_port", "dst_port_number") or [_number(row, "dst_port", "destination_port", "dst_port_number")]) if value > 0]
        source_ports = [value for row in rows for value in _values(row, "src_port", "source_port") if value > 0]
        ports.extend(source_ports)
        source_bytes = sum(_number(row, "src_bytes", "forward_bytes", "sbytes", "totlen_fwd_pkts") for row in rows)
        destination_bytes = sum(_number(row, "dst_bytes", "backward_bytes", "dbytes", "totlen_bwd_pkts") for row in rows)
        source_packets = sum(_number(row, "src_packets", "forward_packets", "spkts", "tot_fwd_pkts") for row in rows)
        destination_packets = sum(_number(row, "dst_packets", "backward_packets", "dpkts", "tot_bwd_pkts") for row in rows)
        sorted_ports = sorted(set(int(port) for port in ports))
        sequential = 0.0
        if len(sorted_ports) > 1:
            sequential = sum(b - a == 1 for a, b in zip(sorted_ports, sorted_ports[1:])) / (len(sorted_ports) - 1)
        flag_aliases = {
            "syn_count": ("syn_count", "tcp_syn_count", "syn_flag_count"),
            "ack_count": ("ack_count", "tcp_ack_count", "ack_flag_count"),
            "fin_count": ("fin_count", "tcp_fin_count", "fin_flag_count"),
            "rst_count": ("rst_count", "tcp_rst_count", "rst_flag_count"),
            "psh_count": ("psh_count", "tcp_psh_count", "fwd_psh_flags"),
            "urg_count": ("urg_count", "tcp_urg_count", "urg_flag_count"),
        }
        flags = {
            name: sum(_number(row, *aliases) for row in rows)
            for name, aliases in flag_aliases.items()
        }
        duration = max((_number(row, "flow_duration", "duration", "dur", "flow_duration_us") for row in rows), default=0.0)
        values = [
            sum(bytes_values),
            sum(packet_values),
            duration,
            _mean(iats),
            _variance(iats),
            max(iats, default=0.0),
            _ratio(source_bytes, destination_bytes),
            _ratio(source_packets, destination_packets),
            flags["syn_count"],
            flags["ack_count"],
            flags["fin_count"],
            flags["rst_count"],
            flags["psh_count"],
            flags["urg_count"],
            _mean(ttl),
            _variance(ttl),
            _mean(tcp_windows),
            _variance(tcp_windows),
            sum(_number(row, "fragment_count", "ip_fragments") for row in rows),
            _mean(payloads),
            _variance(payloads),
            float(len(set(ports))),
            sequential,
            sum(_number(row, "retransmissions", "retransmission_count") for row in rows),
        ]
        return np.asarray(values, dtype=np.float32)

    def aggregate_records(self, events: Iterable[TrafficEvent | Mapping[str, Any]]) -> list[WindowFeatures]:
        """Return one raw feature vector per sliding time window."""

        canonical = [
            event if isinstance(event, TrafficEvent) else TrafficEvent.from_mapping(event)
            for event in events
        ]
        if not canonical:
            return []
        canonical.sort(key=lambda event: event.timestamp)
        start = canonical[0].timestamp
        end = canonical[-1].timestamp
        windows: list[WindowFeatures] = []
        current = start
        while current <= end:
            window_end = current + self.window_seconds
            members = [event for event in canonical if current <= event.timestamp < window_end]
            if members:
                windows.append(
                    WindowFeatures(
                        start=current,
                        end=window_end,
                        values=self._aggregate(members),
                        events=tuple(members),
                    )
                )
            current += self.stride_seconds
        return windows

    def stream_windows(
        self,
        events: Iterable[TrafficEvent | Mapping[str, Any]],
        sample_size: int = 32,
        require_sorted: bool = True,
    ) -> Iterator[WindowFeatures]:
        """Yield bounded-memory windows from timestamp-ordered events.

        Large network captures should be sorted by timestamp. Setting
        ``require_sorted=False`` permits input but may produce incorrect
        boundaries when timestamps move backwards.
        """

        if sample_size < 0:
            raise ValueError("sample_size cannot be negative")
        if self.stride_seconds < self.window_seconds:
            raise ValueError("stream_windows requires stride_seconds >= window_seconds; use batch aggregation for overlapping windows")
        current: _WindowAccumulator | None = None
        last_timestamp: float | None = None
        for item in events:
            event = item if isinstance(item, TrafficEvent) else TrafficEvent.from_mapping(item)
            if last_timestamp is not None and require_sorted and event.timestamp < last_timestamp:
                raise ValueError("stream input is not timestamp sorted; sort it before streaming")
            last_timestamp = event.timestamp
            if current is None:
                current = _WindowAccumulator(event.timestamp, event.timestamp + self.window_seconds, sample_size)
            while event.timestamp >= current.end:
                if current.count:
                    yield current.finish()
                next_start = current.start + self.stride_seconds
                if event.timestamp < next_start:
                    next_start = current.end
                current = _WindowAccumulator(next_start, next_start + self.window_seconds, sample_size)
            current.add(event)
        if current is not None and current.count:
            yield current.finish()

    def aggregate(self, events: Iterable[TrafficEvent | Mapping[str, Any]]) -> np.ndarray:
        """Return one raw feature vector per sliding time window."""

        records = self.aggregate_records(events)
        return (
            np.vstack([record.values for record in records])
            if records
            else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
        )

    def to_sequences(
        self,
        matrix: np.ndarray,
        sequence_length: int,
        normalize: bool = True,
        fit_normalizer: bool = True,
    ) -> np.ndarray:
        """Convert windows to `(batch, sequence_length, feature_dim)` sequences."""

        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected a 2-D matrix with {len(FEATURE_NAMES)} features")
        if normalize:
            prepared = self.normalizer.fit_transform(values) if fit_normalizer else self.normalizer.transform(values)
        else:
            prepared = values
        if len(prepared) < sequence_length:
            return np.empty((0, sequence_length, len(FEATURE_NAMES)), dtype=np.float32)
        return np.stack(
            [prepared[index : index + sequence_length] for index in range(len(prepared) - sequence_length + 1)]
        ).astype(np.float32)
