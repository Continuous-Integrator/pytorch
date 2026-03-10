# mypy: ignore-errors
"""
Convert tile config benchmark CSV data to AutoHeuristic training format.

The AutoHeuristic training pipeline expects a .txt file where:
  - Line 1: JSON metadata (device_capa, shared_memory, name, features)
  - Line 2: CSV header
  - Lines 3+: CSV data rows with features + choice + feedback columns

Input CSV columns (from bench_tile_configs.py):
  M, N, K, config_name, tile_m, tile_n, cluster_m, cluster_n, time_us

Usage:
    python convert_tile_config_data.py \
        --csv /tmp/tile_config_sweep_results/tile_config_feasibility.csv \
        --output async_tp_tile_config_h100_data.txt
"""

import argparse
import csv
import json
import os
import sys


NUMERICAL_FEATURES = [
    "m",
    "k",
    "n",
    "arith_intensity",
    "m_times_k",
    "m_times_n",
    "k_times_n",
]

# H100 device info
H100_SHARED_MEMORY = 232448
H100_DEVICE_CAPA = [9, 0]


def compute_features(m, k, n):
    """Compute derived features for a given (M, K, N) shape."""
    m_times_k = m * k
    m_times_n = m * n
    k_times_n = k * n
    denom = m_times_k + k_times_n + m_times_n
    arith_intensity = round(m * k * n / denom, 4) if denom > 0 else 0
    return {
        "m": m,
        "k": k,
        "n": n,
        "arith_intensity": arith_intensity,
        "m_times_k": m_times_k,
        "m_times_n": m_times_n,
        "k_times_n": k_times_n,
    }


def load_tile_config_csv(csv_path):
    """Load tile config benchmark CSV and convert to (features, choice, feedback) rows.

    Expected CSV columns: M, N, K, config_name, tile_m, tile_n, cluster_m, cluster_n, time_us
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["M"])
            n = int(row["N"])
            k = int(row["K"])
            config_name = row["config_name"]
            time_us = float(row["time_us"])

            feats = compute_features(m, k, n)
            rows.append({**feats, "choice": config_name, "feedback": time_us})
    return rows


def deduplicate_rows(rows):
    """
    Deduplicate by (m, k, n, choice) — keep the row with the lowest feedback (best timing).
    AutoHeuristic expects one feedback value per (features, choice) combination.
    """
    seen = {}
    for row in rows:
        key = (row["m"], row["k"], row["n"], row["choice"])
        if key not in seen or row["feedback"] < seen[key]["feedback"]:
            seen[key] = row
    return list(seen.values())


def write_autoheuristic_txt(rows, output_path, shared_memory, device_capa):
    """
    Write data in AutoHeuristic format:
      Line 1: JSON metadata
      Line 2: CSV header
      Lines 3+: CSV data
    """
    metadata = {
        "shared_memory": shared_memory,
        "device_capa": device_capa,
        "name": "async_tp_tile_config",
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": [],
    }

    header_fields = NUMERICAL_FEATURES + ["choice", "feedback"]

    with open(output_path, "w") as f:
        f.write(json.dumps(metadata) + "\n")
        f.write(",".join(header_fields) + "\n")
        for row in rows:
            values = [str(row[field]) for field in header_fields]
            f.write(",".join(values) + "\n")

    unique_configs = len({(r["m"], r["k"], r["n"]) for r in rows})
    unique_choices = sorted(set(r["choice"] for r in rows))
    print(f"Written: {output_path}")
    print(f"  {len(rows)} rows, {unique_configs} unique (M,K,N) configs")
    print(f"  Choices: {unique_choices}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert tile config benchmark CSV to AutoHeuristic training format"
    )
    parser.add_argument(
        "--csv",
        nargs="+",
        required=True,
        help="Path(s) to tile config benchmark CSV file(s)",
    )
    parser.add_argument(
        "--output",
        default="async_tp_tile_config_h100_data.txt",
        help="Output .txt file in AutoHeuristic format",
    )
    parser.add_argument(
        "--shared-memory",
        type=int,
        default=H100_SHARED_MEMORY,
        help="GPU shared memory size (bytes)",
    )
    parser.add_argument(
        "--device-capa",
        type=str,
        default="9,0",
        help="Device capability, comma-separated (e.g., '9,0' for H100)",
    )
    args = parser.parse_args()

    device_capa = [int(x) for x in args.device_capa.split(",")]

    all_rows = []
    for csv_path in args.csv:
        if not os.path.exists(csv_path):
            print(f"  CSV not found: {csv_path}", file=sys.stderr)
            continue
        print(f"Loading: {csv_path}")
        csv_rows = load_tile_config_csv(csv_path)
        all_rows.extend(csv_rows)
        print(f"  {len(csv_rows)} rows")

    if not all_rows:
        print("ERROR: No data found.", file=sys.stderr)
        sys.exit(1)

    all_rows = deduplicate_rows(all_rows)
    print(f"\nAfter dedup: {len(all_rows)} rows")

    write_autoheuristic_txt(all_rows, args.output, args.shared_memory, device_capa)


if __name__ == "__main__":
    main()
