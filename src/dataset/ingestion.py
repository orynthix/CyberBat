"""Offline flow CSV and optional PCAP ingestion adapters."""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ..config import ATTACK_CATEGORY_TO_STAGE
from .feature_extractor import TrafficEvent


def read_flow_csv(path: str | Path) -> Iterator[TrafficEvent]:
    """Yield canonical events from a CSV file without third-party dependencies."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"flow CSV does not exist: {source}")
    with source.open("r", newline="", encoding="utf-8-sig") as handle:
        delimiter = "\t" if source.suffix.lower() == ".tsv" else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames:
            raise ValueError(f"flow CSV has no header: {source}")
        for row_number, row in enumerate(reader, start=2):
            if not any(value not in (None, "") for value in row.values()):
                continue
            try:
                yield TrafficEvent.from_mapping(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid flow row {row_number} in {source}: {exc}") from exc


def _parquet_row_to_event(row: Mapping[str, Any], row_index: int) -> TrafficEvent:
    """Normalize common Kaggle Parquet datasets into the event schema."""

    values = dict(row)
    if not any(values.get(name) not in (None, "") for name in ("timestamp", "ts", "time", "Timestamp")):
        values["timestamp"] = float(row_index)
    label = values.get("label", values.get("Label", values.get("ClassLabel", 0)))
    attack_category = values.get("attack_cat", values.get("Family", ""))
    label_text = str(label).strip().lower()
    category_text = str(attack_category).strip().lower()
    category_stage = ATTACK_CATEGORY_TO_STAGE.get(category_text)
    label_is_benign = label_text in {"0", "0.0", "benign", "normal", "background"}
    is_benign = label_is_benign or category_stage == 0
    values["risk"] = 0.0 if is_benign else 1.0
    values["stage"] = 0 if is_benign else (category_stage if category_stage is not None else 1)
    return TrafficEvent.from_mapping(values)


def read_parquet(path: str | Path, batch_size: int = 65_536) -> Iterator[TrafficEvent]:
    """Stream Parquet record batches through PyArrow without full-file loading."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Parquet file does not exist: {source}")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Parquet ingestion requires the 'pyarrow' package") from exc
    row_index = 0
    parquet_file = parquet.ParquetFile(source)
    for batch in parquet_file.iter_batches(batch_size=batch_size, use_threads=True):
        for row in batch.to_pylist():
            yield _parquet_row_to_event(row, row_index)
            row_index += 1


def read_pcap(path: str | Path) -> Iterator[TrafficEvent]:
    """Yield packet events using Scapy when the optional PCAP dependency exists."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PCAP file does not exist: {source}")
    try:
        from scapy.all import IP, TCP, UDP, PcapReader
    except ImportError as exc:
        raise RuntimeError(
            "PCAP ingestion requires the optional 'scapy' package; install it for PCAP input"
        ) from exc
    with PcapReader(str(source)) as packets:
        for packet in packets:
            yield from _packet_to_event(packet, IP, TCP, UDP)


def _packet_to_event(packet: Any, ip_layer: Any, tcp_layer: Any, udp_layer: Any) -> Iterator[TrafficEvent]:
    """Convert one Scapy packet without retaining the capture in memory."""
    if not packet.haslayer(ip_layer):
        return
    ip = packet[ip_layer]
    transport = packet.getlayer(tcp_layer) or packet.getlayer(udp_layer)
    values: dict[str, Any] = {
        "timestamp": float(packet.time),
        "bytes": len(packet),
        "packets": 1,
        "ttl": getattr(ip, "ttl", 0),
        "fragment_count": int(bool(getattr(ip, "flags", 0) and getattr(ip, "frag", 0))),
        "payload_size": len(bytes(getattr(ip, "payload", b""))),
        "src_ip": getattr(ip, "src", ""),
        "dst_ip": getattr(ip, "dst", ""),
    }
    if transport is not None:
        values.update(
            {
                "src_port": getattr(transport, "sport", 0),
                "dst_port": getattr(transport, "dport", 0),
                "tcp_window": getattr(transport, "window", 0),
            }
        )
        flags = str(getattr(transport, "flags", ""))
        for flag_name, flag in (("syn_count", "S"), ("ack_count", "A"), ("fin_count", "F"), ("rst_count", "R"), ("psh_count", "P"), ("urg_count", "U")):
            values[flag_name] = int(flag in flags)
    yield TrafficEvent.from_mapping(values)


def load_events(path: str | Path) -> Iterator[TrafficEvent]:
    """Select an ingestion adapter from the file extension."""

    source_path = Path(path)
    if source_path.is_dir():
        files = sorted(
            file
            for file in source_path.rglob("*")
            if file.suffix.lower() in {".csv", ".tsv", ".parquet", ".pq", ".pcap", ".pcapng"}
        )
        if not files:
            raise ValueError(f"no supported traffic files found in directory: {source_path}")
        row_index = 0
        for file in files:
            for event in load_events(file):
                values = dict(event.values)
                values["timestamp"] = float(row_index)
                yield TrafficEvent.from_mapping(values)
                row_index += 1
        return
    suffix = source_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        yield from read_flow_csv(path)
    elif suffix in {".parquet", ".pq"}:
        yield from read_parquet(path)
    elif suffix in {".pcap", ".pcapng"}:
        yield from read_pcap(path)
    else:
        raise ValueError(f"unsupported traffic file extension: {suffix or '<none>'}")
