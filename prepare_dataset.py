"""Profile huge traffic files and create compact local caches."""

from __future__ import annotations

import argparse

from src.dataset.large_files import build_compact_cache, profile_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV, TSV, PCAP, or PCAPNG path")
    parser.add_argument("--output", default="data/cache", help="Compact cache directory")
    args = parser.parse_args()
    report_path = f"{args.output}/dataset_report.json"
    report = profile_dataset(args.input, report_path)
    metadata = build_compact_cache(args.input, args.output)
    print(f"Profiled {report['event_count']} events into {report['window_count']} windows.")
    print(
        f"Compact cache: {metadata['window_count']} x {metadata['feature_count']} "
        f"{metadata['features_dtype']} normalized features."
    )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
