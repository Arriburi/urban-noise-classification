"""
AI-generated analysis utility -- not part of the core simulation logic.

Aggregates groundtruth hit/miss CSVs from hit_results/ across all 30 seeds.

Outputs (saved to hit_results/):
  - hit_summary.csv          Per-class avg hit rate + base dataset counts
  - miss_flow_matrix.csv     Where do misses go? (target x actual)
  - cost_of_balancing.csv    Effective steps spent per actual class gain

Run:
    uv run clap-env/simulation/analyze_hits.py
"""

import os
from collections import Counter

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
HIT_DIR = os.path.join(BASE_DIR, "hit_results")
BASE_PARQUET = os.path.join(BASE_DIR, "audioset_eval_top_mixed_no_mixed.parquet")

CLASS_NAMES = [
    "Human sounds",
    "Animal",
    "Music",
    "Natural sounds",
    "Source-ambiguous sounds",
    "Channel, environment and background",
    "Sounds of things",
]


# ---------------------------------------------------------------------------
# 1. Load all hit CSVs and collect raw (target, actual) pairs per seed
# ---------------------------------------------------------------------------
def load_hit_logs() -> list[list[tuple[str, str]]]:
    logs = []
    for fname in sorted(os.listdir(HIT_DIR)):
        if not fname.endswith(".csv") or "_hits_" not in fname:
            continue
        df = pd.read_csv(os.path.join(HIT_DIR, fname))
        pairs = []
        for _, row in df.iterrows():
            target = row["target_class"]
            if target == "OVERALL":
                continue
            # Hits: target == actual
            for _ in range(int(row["hits"])):
                pairs.append((target, target))
            # Misses: parse miss_detail "Sounds of things: 20; Human sounds: 13"
            detail = str(row["miss_detail"]).strip()
            if detail and detail != "-":
                for chunk in detail.split(";"):
                    chunk = chunk.strip()
                    if not chunk:
                        continue
                    # last token after last colon is the count
                    parts = chunk.rsplit(":", 1)
                    actual_class = parts[0].strip()
                    count = int(parts[1].strip())
                    for _ in range(count):
                        pairs.append((target, actual_class))
        logs.append(pairs)
    return logs


# ---------------------------------------------------------------------------
# 2. Base dataset class distribution
# ---------------------------------------------------------------------------
def load_base_distribution() -> dict[str, int]:
    df = pd.read_parquet(BASE_PARQUET)
    counts = Counter()
    for labels in df["human_labels"]:
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            for lab in labels:
                counts[lab] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# 3. Per-class summary: avg hit rate across seeds + base counts
# ---------------------------------------------------------------------------
def build_hit_summary(
    logs: list[list[tuple[str, str]]], base_counts: dict[str, int]
) -> pd.DataFrame:
    base_total = sum(base_counts.get(c, 0) for c in CLASS_NAMES)
    rows = []

    for cls in CLASS_NAMES:
        hit_rates = []
        totals = []
        for log in logs:
            entries = [(t, a) for t, a in log if t == cls]
            total = len(entries)
            hits = sum(1 for t, a in entries if t == a)
            if total > 0:
                hit_rates.append(hits / total)
                totals.append(total)

        base_n = base_counts.get(cls, 0)
        base_pct = base_n / base_total * 100 if base_total > 0 else 0

        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0
        std_hit_rate = np.std(hit_rates, ddof=1) if len(hit_rates) > 1 else 0
        avg_attempts = np.mean(totals) if totals else 0

        rows.append(
            {
                "class": cls,
                "base_count": base_n,
                "base_pct": round(base_pct, 2),
                "avg_attempts": round(avg_attempts, 1),
                "avg_hit_rate": round(avg_hit_rate, 4),
                "std_hit_rate": round(std_hit_rate, 4),
                "seeds": len(hit_rates),
            }
        )

    return pd.DataFrame(rows).sort_values("avg_hit_rate").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Miss-flow matrix: when targeting class X, what class did we get?
# ---------------------------------------------------------------------------
def build_miss_flow(logs: list[list[tuple[str, str]]]) -> pd.DataFrame:
    matrix = {t: Counter() for t in CLASS_NAMES}
    for log in logs:
        for target, actual in log:
            if target != actual and target in matrix:
                matrix[target][actual] += 1

    rows = []
    for target in CLASS_NAMES:
        row = {"target": target}
        for actual in CLASS_NAMES:
            row[actual] = matrix[target].get(actual, 0)
        rows.append(row)

    return pd.DataFrame(rows).set_index("target")


# ---------------------------------------------------------------------------
# 5. Cost of balancing: how many attempts per actual class gain?
# ---------------------------------------------------------------------------
def build_cost_table(logs: list[list[tuple[str, str]]]) -> pd.DataFrame:
    rows = []
    for cls in CLASS_NAMES:
        costs = []
        for log in logs:
            entries = [(t, a) for t, a in log if t == cls]
            total = len(entries)
            hits = sum(1 for t, a in entries if t == a)
            if hits > 0:
                costs.append(total / hits)
            elif total > 0:
                costs.append(float("inf"))

        avg_cost = np.mean([c for c in costs if c != float("inf")]) if costs else 0
        inf_seeds = sum(1 for c in costs if c == float("inf"))

        rows.append(
            {
                "class": cls,
                "avg_cost_per_hit": (
                    round(avg_cost, 2) if avg_cost != float("inf") else "inf"
                ),
                "seeds_with_zero_hits": inf_seeds,
                "interpretation": (
                    f"~{round(avg_cost, 1)} attempts needed to gain 1 sample"
                    if avg_cost != float("inf") and avg_cost > 0
                    else "no data"
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "avg_cost_per_hit",
            key=lambda x: pd.to_numeric(x, errors="coerce").fillna(float("inf")),
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Loading hit CSVs from: {HIT_DIR}")
    logs = load_hit_logs()
    print(f"Loaded {len(logs)} seed files")

    print(f"Loading base dataset: {BASE_PARQUET}")
    base_counts = load_base_distribution()

    # 1) Hit summary with base distribution
    summary = build_hit_summary(logs, base_counts)
    print("\n=== Hit/Miss Summary (sorted by worst hit rate) ===")
    print(summary.to_string(index=False))

    # 2) Miss flow matrix
    miss_flow = build_miss_flow(logs)
    print("\n=== Miss Flow Matrix (target rows -> actual columns) ===")
    print(miss_flow.to_string())

    # 3) Cost of balancing
    cost = build_cost_table(logs)
    print("\n=== Cost of Balancing ===")
    print(cost.to_string(index=False))

    # Save
    summary.to_csv(os.path.join(HIT_DIR, "hit_summary.csv"), index=False)
    miss_flow.to_csv(os.path.join(HIT_DIR, "miss_flow_matrix.csv"))
    cost.to_csv(os.path.join(HIT_DIR, "cost_of_balancing.csv"), index=False)

    print(f"\nSaved to {HIT_DIR}:")
    print("  - hit_summary.csv")
    print("  - miss_flow_matrix.csv")
    print("  - cost_of_balancing.csv")


if __name__ == "__main__":
    main()
