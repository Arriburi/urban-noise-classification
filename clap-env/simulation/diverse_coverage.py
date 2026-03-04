import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BASE_PARQUET = os.path.join(BASE_DIR, "audioset_eval_mid.parquet")

def parse_diverse_filename(name: str) -> Tuple[int, int] | None:
    """
    Match patterns like:
      audioset_eval_mid_diverse_2000_seed1.parquet
      mid_diverse_2000_seed1.parquet
    Returns (steps, seed) or None.
    """
    m = re.match(r".*diverse_(\d+)_seed(\d+)\.parquet$", name)
    if not m:
        return None
    steps = int(m.group(1))
    seed = int(m.group(2))
    return steps, seed


def load_base_labels() -> List[str]:
    """Load full label universe from the base mid parquet."""
    if not os.path.exists(BASE_PARQUET):
        raise FileNotFoundError(f"Base parquet not found: {BASE_PARQUET}")

    df = pd.read_parquet(BASE_PARQUET)
    labels_set = set()
    for labels in df["human_labels"]:
        if isinstance(labels, (list, tuple, np.ndarray)):
            labels_set.update(labels)
    return sorted(labels_set)


def compute_coverage_for_seed(
    path: str, seed: int, all_labels: List[str]
) -> pd.DataFrame:
    """Return per-step coverage (unique labels found) for a single seed."""
    df = pd.read_parquet(path)
    label_universe = set(all_labels)

    seen = set()
    records = []

    for step, labels in enumerate(df["human_labels"], start=1):
        if isinstance(labels, (list, tuple, np.ndarray)):
            for lab in labels:
                if lab in label_universe:
                    seen.add(lab)

        records.append(
            {
                "seed": seed,
                "step": step,
                "unique_labels_found": len(seen),
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Outputs directory not found: {OUTPUT_DIR}")
        return

    files = sorted(
        f for f in os.listdir(OUTPUT_DIR) if f.endswith(".parquet") and "diverse" in f
    )
    if not files:
        print(f"No diverse parquet files found in {OUTPUT_DIR}")
        return

    print(f"Loading base labels from: {BASE_PARQUET}")
    all_labels = load_base_labels()
    total_labels = len(all_labels) or 1
    print(f"Found {total_labels} unique labels in base dataset")

    # Parse file metadata first so we can ignore short test runs (e.g. 200-step seed1)
    parsed_files = []
    for fname in files:
        parsed = parse_diverse_filename(fname)
        if parsed is None:
            print(f"Skipping non-diverse file: {fname}")
            continue
        steps, seed = parsed
        parsed_files.append((fname, steps, seed))

    if not parsed_files:
        print("No diverse files with recognizable naming found.")
        return

    # Keep only runs with the maximum number of steps (e.g. 2000), so
    # earlier smoke-test runs (like 200 steps) don't skew the averages.
    max_steps = max(steps for _, steps, _ in parsed_files)
    parsed_files = [(f, s, seed) for (f, s, seed) in parsed_files if s == max_steps]

    all_coverage: List[pd.DataFrame] = []

    print(f"Using {len(parsed_files)} diverse output files (steps={max_steps}) from {OUTPUT_DIR}")
    for fname, steps, seed in parsed_files:
        path = os.path.join(OUTPUT_DIR, fname)
        print(f"Processing seed {seed} from {fname} (steps={steps})")

        coverage_df = compute_coverage_for_seed(path, seed, all_labels)
        all_coverage.append(coverage_df)

    if not all_coverage:
        print("No coverage data computed.")
        return

    coverage_all_df = pd.concat(all_coverage, ignore_index=True)

    steps_of_interest = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]

    summary_rows = []
    for step in steps_of_interest:
        sub = coverage_all_df[coverage_all_df["step"] == step]
        if sub.empty:
            continue
        mean_classes = sub["unique_labels_found"].mean()
        std_classes = sub["unique_labels_found"].std(ddof=1) if len(sub) > 1 else 0.0
        pct = mean_classes / total_labels * 100.0
        summary_rows.append(
            {
                "step": int(step),
                "mean_classes": mean_classes,
                "pct_of_labels": pct,
                "total_labels": total_labels,
                "std_classes": std_classes,
                "num_seeds": len(sub),
            }
        )

    if summary_rows:
        summary_df = (
            pd.DataFrame(summary_rows)
            .sort_values("step")
            .reset_index(drop=True)
        )
        # Pretty-print in the requested format
        label_header = f"% of {total_labels} labels"
        print(f"\n  Step       Mean classes   {label_header:>14s}     Std")
        print("  " + "-" * (8 + 14 + 16 + 8))
        for _, row in summary_df.iterrows():
            step = int(row["step"])
            mean_c = row["mean_classes"]
            pct = row["pct_of_labels"]
            std_c = row["std_classes"]
            print(
                f"{step:6d}  "
                f"{mean_c:14.1f}  "
                f"{pct:14.1f}  "
                f"{std_c:8.1f}"
            )

if __name__ == "__main__":
    main()

