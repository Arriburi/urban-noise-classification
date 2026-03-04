import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "outputs"
RESULTS_DIR = SCRIPT_DIR / "distribution_results"

# --- Config: edit these to change base parquet, simulations, and output names ---
BASE_PARQUET = SCRIPT_DIR / "audioset_eval_mid.parquet"
# Hybrid config: varying total steps, varying diverse steps.
# Filenames like: mid_hybrid_2000_d300_n15_seed1.parquet, mid_hybrid_3000_d1000_n15_seed2.parquet, ...
# We group results by the DIVERSE phase length (the "dXXX" part).
SIM_GLOB = "mid_hybrid_*_d*_n15_seed*.parquet"
PARSE_PATTERN = re.compile(r"^mid_hybrid_\d+_d(\d+)_n15_seed(\d+)\.parquet$")
PLOTS_METHOD_NAME = "hybrid_d"  # Outputs hybrid_d300, hybrid_d1000, ...

# Previous diverse-only config (kept here for reference):
# SIM_GLOB = "mid_diverse_*.parquet"
# PARSE_PATTERN = re.compile(r"^mid_diverse_(\d+)_seed(\d+)\.parquet$")
# PLOTS_METHOD_NAME = "diverse"  # Used for {method}.csv and {method}_summary.csv (plots.py expects: diverse, random, groundtruth_15, mean_250, similar)


def get_distribution(file_path: Path) -> tuple[pd.Series, int]:
    df = pd.read_parquet(file_path)
    labels_flat = []
    for labels in df["human_labels"]:
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            labels_flat.extend(labels)
    counts = pd.Series(labels_flat).value_counts()
    return counts, len(df)


def pct_vector(counts: pd.Series, total: int, all_labels: list[str]) -> np.ndarray:
    out = np.zeros(len(all_labels))
    if total <= 0:
        return out
    for i, label in enumerate(all_labels):
        out[i] = counts.get(label, 0) / total * 100.0
    return out


def imbalance(pct: np.ndarray, n_classes: int) -> float:
    uniform_pct = 100.0 / n_classes if n_classes > 0 else 0.0
    return float(np.mean(np.abs(pct - uniform_pct)))


def shift(pct: np.ndarray, base_pct: np.ndarray) -> float:
    return float(np.mean(np.abs(pct - base_pct)))


def compute_entropy(pct: np.ndarray, n_classes: int) -> float:
    s = pct.sum()
    probs = (pct / s) if s > 0 else np.ones(n_classes) / n_classes
    return float(scipy_entropy(probs, base=2))


def parse_simulation_name(name: str, pattern: re.Pattern) -> tuple[int | None, int | None]:
    """Return (steps, seed) or (None, None). Pattern must have two groups: steps, seed."""
    m = pattern.match(name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    if not BASE_PARQUET.exists():
        print(f"Base parquet not found: {BASE_PARQUET}")
        return
    if not OUTPUTS_DIR.exists():
        print(f"Outputs dir not found: {OUTPUTS_DIR}")
        return

    if RESULTS_DIR.exists():
        shutil.rmtree(RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    base_counts, base_total = get_distribution(BASE_PARQUET)
    all_labels = sorted(base_counts.index.tolist())
    n_classes = len(all_labels)
    base_pct = pct_vector(base_counts, base_total, all_labels)
    base_entropy = compute_entropy(base_pct, n_classes)
    base_imbalance = imbalance(base_pct, n_classes)

    # Labels ordered by count descending (for readable bar charts)
    labels_by_count = sorted(
        all_labels,
        key=lambda L: base_counts.get(L, 0),
        reverse=True,
    )

    # --- base.csv: rows = labels + ENTROPY, IMBALANCE, SHIFT; column "base" ---
    # Format matches old git: class rows "c (p%)", metrics ".3f" strings
    base_data = {}
    for label in labels_by_count:
        c = base_counts.get(label, 0)
        p = (c / base_total * 100.0) if base_total > 0 else 0.0
        base_data[label] = f"{c} ({p:.1f}%)"
    base_data["ENTROPY"] = f"{base_entropy:.3f}"
    base_data["IMBALANCE"] = f"{base_imbalance:.3f}"
    base_data["SHIFT"] = "0.000"  # base has no shift to itself
    base_df = pd.DataFrame({"base": base_data})
    base_df.to_csv(RESULTS_DIR / "base.csv")
    print("base -> base.csv")

    # Group simulations by (diverse) steps
    by_steps: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(OUTPUTS_DIR.glob(SIM_GLOB)):
        steps, seed = parse_simulation_name(path.name, PARSE_PATTERN)
        if steps is None or seed is None:
            continue
        if steps not in by_steps:
            by_steps[steps] = []
        by_steps[steps].append((seed, path))

    if not by_steps:
        print("No simulation parquets found.")
        return

    # For hybrid, "steps" here is the diverse phase length (e.g. 300/400/500/750)
    for steps in sorted(by_steps.keys()):
        entries = sorted(by_steps[steps], key=lambda x: x[0])
        method_name = f"{PLOTS_METHOD_NAME}{steps}"

        # Build method CSV: rows = labels + ENTROPY, IMBALANCE, SHIFT; columns = seed1, seed2, ...
        method_cols = {}
        pct_arrays = []  # for computing mean per label
        entropies = []

        for seed, path in entries:
            col_name = f"seed{seed}"
            counts, recordings = get_distribution(path)
            pct = pct_vector(counts, recordings, all_labels)
            pct_arrays.append(pct)
            ent = compute_entropy(pct, n_classes)
            imb = imbalance(pct, n_classes)
            shf = shift(pct, base_pct)
            entropies.append(ent)

            col_data = {}
            for label in labels_by_count:
                idx = all_labels.index(label)
                c = counts.get(all_labels[idx], 0)
                p = pct[idx]
                col_data[label] = f"{c} ({p:.1f}%)"
            col_data["ENTROPY"] = f"{ent:.3f}"
            col_data["IMBALANCE"] = f"{imb:.3f}"
            col_data["SHIFT"] = f"{shf:.3f}"
            method_cols[col_name] = col_data

        method_df = pd.DataFrame(method_cols)
        method_df.to_csv(RESULTS_DIR / f"{method_name}.csv")
        print(f"{method_name} ({steps} diverse steps, {len(entries)} seeds) -> {method_name}.csv")
        print(f"  entropy: {[round(e, 4) for e in entropies]}")

        pct_stack = np.stack(pct_arrays)
        # --- {method}_summary.csv: full old-format (mean, std, mean-std, mean+std, min, max) ---
        ordered_cols = dict(sorted(method_cols.items(), key=lambda x: int(x[0].replace("seed", ""))))
        metric_arrays = {m: [] for m in ["ENTROPY", "IMBALANCE", "SHIFT"]}
        for col_name, col_data in ordered_cols.items():
            for m in metric_arrays:
                metric_arrays[m].append(float(col_data[m]))

        summary_rows = []
        for label in labels_by_count:
            idx = all_labels.index(label)
            pcts = pct_stack[:, idx]
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
        for m in ["ENTROPY", "IMBALANCE", "SHIFT"]:
            arr = np.array(metric_arrays[m])
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
            summary_rows.append({
                "metric": m,
                "mean": f"{mean_val:.4f}",
                "std": f"{std_val:.4f}",
                "mean-std": f"{mean_val - std_val:.4f}",
                "mean+std": f"{mean_val + std_val:.4f}",
                "min": f"{float(np.min(arr)):.4f}",
                "max": f"{float(np.max(arr)):.4f}",
            })
        summary_df = pd.DataFrame(summary_rows).set_index("metric")
        summary_df.to_csv(RESULTS_DIR / f"{method_name}_summary.csv")
        print(f"{method_name}: summary -> {method_name}_summary.csv")

    print(f"\nResults in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
