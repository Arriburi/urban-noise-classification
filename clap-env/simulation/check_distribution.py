"""
One CSV per mode. Base written to base.csv. Config: which window for groundtruth and mean.
"""
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# Which window to use for groundtruth and mean (all other windows ignored).
GT_WINDOW = 15
MEAN_WINDOW = 250

CLASS_NAMES = [
    "Human sounds",
    "Animal",
    "Music",
    "Natural sounds",
    "Source-ambiguous sounds",
    "Channel, environment and background",
    "Sounds of things",
]
C = len(CLASS_NAMES)
UNIFORM_PCT = 100.0 / C

MODES = ["random", "similar", "diverse", "groundtruth", "mean"]


def get_distribution(file_path: Path) -> tuple[pd.Series, int]:
    df = pd.read_parquet(file_path)
    labels_flat = []
    for labels in df["human_labels"]:
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            labels_flat.extend(labels)
    counts = pd.Series(labels_flat).value_counts()
    return counts, len(df)


def pct_vector(counts: pd.Series, total: int) -> np.ndarray:
    out = np.zeros(C)
    if total <= 0:
        return out
    for i, label in enumerate(CLASS_NAMES):
        out[i] = counts.get(label, 0) / total * 100.0
    return out


def imbalance(pct: np.ndarray) -> float:
    return float(np.mean(np.abs(pct - UNIFORM_PCT)))


def shift(pct: np.ndarray, base_pct: np.ndarray) -> float:
    return float(np.mean(np.abs(pct - base_pct)))


def parse_parquet_name(name: str) -> tuple[str | None, int | None, int | None]:
    """Return (mode, window, seed). mode in MODES; window None for random/similar/diverse."""
    name = name.replace("top_mixed_no_mixed_", "").replace(".parquet", "")
    # groundtruth_n15_1575_seed3 -> groundtruth, 15, 3
    # random_1575_seed2 -> random, None, 2
    seed_m = re.search(r"seed(\d+)$", name)
    seed = int(seed_m.group(1)) if seed_m else None
    window = None
    if "groundtruth" in name or "mean" in name:
        w_m = re.search(r"n(\d+)", name)
        window = int(w_m.group(1)) if w_m else None
    if "groundtruth" in name:
        return ("groundtruth", window, seed)
    if "mean" in name:
        return ("mean", window, seed)
    if name.startswith("random"):
        return ("random", None, seed)
    if name.startswith("similar"):
        return ("similar", None, seed)
    if name.startswith("diverse"):
        return ("diverse", None, seed)
    return (None, None, None)


def build_table(columns: dict[str, tuple[pd.Series, int]], base_pct: np.ndarray) -> pd.DataFrame:
    """columns: seed1 -> (counts, total), etc. Rows: CLASS_NAMES + ENTROPY, IMBALANCE, SHIFT."""
    rows = []
    for label in CLASS_NAMES:
        row = {"label": label}
        for col_name, (counts, total) in columns.items():
            c = counts.get(label, 0)
            p = (c / total * 100.0) if total > 0 else 0.0
            row[col_name] = f"{c} ({p:.1f}%)"
        rows.append(row)
    ent_row = {"label": "ENTROPY"}
    imb_row = {"label": "IMBALANCE"}
    shf_row = {"label": "SHIFT"}
    for col_name, (counts, total) in columns.items():
        pct = pct_vector(counts, total)
        s = pct.sum()
        probs = (pct / s) if s > 0 else np.ones(C) / C
        ent_row[col_name] = f"{scipy_entropy(probs, base=2):.3f}"
        imb_row[col_name] = f"{imbalance(pct):.3f}"
        shf_row[col_name] = f"{shift(pct, base_pct):.3f}"
    rows.extend([ent_row, imb_row, shf_row])
    return pd.DataFrame(rows).set_index("label")


def main():
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    results_dir = script_dir / "distribution_results"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = script_dir / "audioset_eval_top_mixed_no_mixed.parquet"
    base_counts, base_total = get_distribution(base_path)
    base_pct = pct_vector(base_counts, base_total)

    # One dict per mode: column name -> (counts, total). Column name = seed{N}.
    by_mode: dict[str, dict[str, tuple[pd.Series, int]]] = {m: {} for m in MODES}

    for path in sorted(outputs_dir.glob("top_mixed_no_mixed_*.parquet")):
        mode, window, seed = parse_parquet_name(path.name)
        if mode is None or seed is None:
            continue
        if mode == "groundtruth" and window != GT_WINDOW:
            continue
        if mode == "mean" and window != MEAN_WINDOW:
            continue
        col = f"seed{seed}"
        by_mode[mode][col] = get_distribution(path)

    # Base
    build_table({"base": (base_counts, base_total)}, base_pct).to_csv(results_dir / "base.csv")
    print("base -> base.csv")

    # One CSV per mode, columns sorted seed1..seed30
    mode_csv_files = {}
    for mode in MODES:
        cols = by_mode[mode]
        if not cols:
            continue
        ordered = dict(sorted(cols.items(), key=lambda x: int(x[0].replace("seed", ""))))
        if mode == "groundtruth":
            out_name = f"groundtruth_{GT_WINDOW}.csv"
        elif mode == "mean":
            out_name = f"mean_{MEAN_WINDOW}.csv"
        else:
            out_name = f"{mode}.csv"
        csv_path = results_dir / out_name
        build_table(ordered, base_pct).to_csv(csv_path)
        mode_csv_files[mode] = csv_path
        print(f"{mode}: {len(ordered)} runs -> {out_name}")

    # Generate summary files: class average % (from exact counts) + mean ± std for ENTROPY, IMBALANCE, SHIFT
    for mode in MODES:
        if mode not in mode_csv_files:
            continue
        csv_path = mode_csv_files[mode]
        ordered = dict(sorted(by_mode[mode].items(), key=lambda x: int(x[0].replace("seed", ""))))
        df = pd.read_csv(csv_path, index_col=0)
        metrics = ["ENTROPY", "IMBALANCE", "SHIFT"]
        missing_metrics = [m for m in metrics if m not in df.index]
        if missing_metrics:
            print(f"Warning: {mode} CSV missing metrics: {missing_metrics}. Skipping summary.")
            continue

        # Class average % from exact counts (no parsing)
        summary_rows = []
        for label in CLASS_NAMES:
            pcts = []
            for (counts, total) in ordered.values():
                if total > 0:
                    pcts.append(counts.get(label, 0) / total * 100.0)
                else:
                    pcts.append(0.0)
            arr = np.array(pcts)
            mean_pct = float(np.mean(arr))
            std_pct = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            summary_rows.append({
                "metric": label,
                "mean": f"{mean_pct:.4f}",
                "std": f"{std_pct:.4f}",
                "mean-std": f"{mean_pct - std_pct:.4f}",
                "mean+std": f"{mean_pct + std_pct:.4f}",
                "min": f"{float(np.min(arr)):.4f}",
                "max": f"{float(np.max(arr)):.4f}",
            })

        for metric in metrics:
            values = []
            for col in df.columns:
                if col.startswith("seed"):
                    try:
                        values.append(float(df.loc[metric, col]))
                    except (ValueError, KeyError):
                        print(f"Warning: {mode} CSV, {metric} row, column {col}: could not parse value.")
                        continue
            if not values:
                print(f"Warning: {mode} CSV, {metric} row: no valid seed values found. Skipping summary.")
                continue
            arr = np.array(values)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1))  # ddof=1: sample std (n-1 divisor, correct for 30 seeds)
            summary_rows.append({
                "metric": metric,
                "mean": f"{mean_val:.4f}",
                "std": f"{std_val:.4f}",
                "mean-std": f"{mean_val - std_val:.4f}",
                "mean+std": f"{mean_val + std_val:.4f}",
                "min": f"{float(np.min(arr)):.4f}",
                "max": f"{float(np.max(arr)):.4f}",
            })
        summary_df = pd.DataFrame(summary_rows).set_index("metric")
        if mode == "groundtruth":
            summary_name = f"groundtruth_{GT_WINDOW}_summary.csv"
        elif mode == "mean":
            summary_name = f"mean_{MEAN_WINDOW}_summary.csv"
        else:
            summary_name = f"{mode}_summary.csv"
        summary_df.to_csv(results_dir / summary_name)
        print(f"{mode}: summary -> {summary_name}")

    print(f"\nResults in: {results_dir}")


if __name__ == "__main__":
    main()
