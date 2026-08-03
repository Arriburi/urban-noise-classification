import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy as scipy_entropy


SCRIPT_DIR = Path(__file__).parent
OUTPUTS_DIR = SCRIPT_DIR / "simulation_outputs"
RESULTS_DIR = SCRIPT_DIR / "distribution_results"
PLOTS_DIR = SCRIPT_DIR / "plots"

BASE_PARQUET = SCRIPT_DIR / "audioset_eval_mid.parquet"

# --- Strategies to process ---
# Each entry: (glob_pattern, parse_regex, method_name)
# Only processes files matching the specified step counts
TARGET_STEPS = {2000}

STRATEGIES = [
    (
        "mid_random_*_seed*.parquet",
        re.compile(r"^mid_random_(\d+)_seed(\d+)\.parquet$"),
        "random",
    ),
    (
        "mid_diverse_*_seed*.parquet",
        re.compile(r"^mid_diverse_(\d+)_seed(\d+)\.parquet$"),
        "diverse",
    ),
    (
        "mid_groundtruth_cold_k0_nb_*_seed*.parquet",
        re.compile(r"^mid_groundtruth_cold_k0_nb_(\d+)_seed(\d+)\.parquet$"),
        "groundtruth_cold_k0_nb",
    ),
    (
        "mid_groundtruth_cold_k0_*_seed*.parquet",
        re.compile(r"^mid_groundtruth_cold_k0_(\d+)_seed(\d+)\.parquet$"),
        "groundtruth_cold_k0",
    ),
    (
        "mid_groundtruth_cold_k2_*_seed*.parquet",
        re.compile(r"^mid_groundtruth_cold_k2_(\d+)_seed(\d+)\.parquet$"),
        "groundtruth_cold_k2",
    ),
    (
        "mid_kmeans_hybrid_km57_k2_*_seed*.parquet",
        re.compile(r"^mid_kmeans_hybrid_km57_k2_(\d+)_seed(\d+)\.parquet$"),
        "kmeans_hybrid_km57",
    ),
    (
        "mid_kmeans_hybrid_km100_k2_*_seed*.parquet",
        re.compile(r"^mid_kmeans_hybrid_km100_k2_(\d+)_seed(\d+)\.parquet$"),
        "kmeans_hybrid_km100",
    ),
    (
        "mid_kmeans_hybrid_km150_k2_*_seed*.parquet",
        re.compile(r"^mid_kmeans_hybrid_km150_k2_(\d+)_seed(\d+)\.parquet$"),
        "kmeans_hybrid_km150",
    ),
]


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
    m = pattern.match(name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def process_strategy(
    sim_glob: str,
    parse_pattern: re.Pattern,
    method_name: str,
    base_counts: pd.Series,
    base_total: int,
    all_labels: list[str],
    labels_by_count: list[str],
    base_pct: np.ndarray,
    n_classes: int,
) -> None:
    # Group files by steps
    by_steps: dict[int, list[tuple[int, Path]]] = {}
    for path in sorted(OUTPUTS_DIR.glob(sim_glob)):
        steps, seed = parse_simulation_name(path.name, parse_pattern)
        if steps is None or seed is None:
            continue
        if steps not in TARGET_STEPS:
            continue
        by_steps.setdefault(steps, []).append((seed, path))

    if not by_steps:
        print(f"  No files found for {method_name}")
        return

    total_files = sum(len(v) for v in by_steps.values())
    print(f"\n--- {method_name} ---")
    print(f"  Found {total_files} files, step configs: {sorted(by_steps.keys())}")

    for steps in sorted(by_steps.keys()):
        entries = sorted(by_steps[steps], key=lambda x: x[0])
        full_name = f"{method_name}_{steps}"

        method_cols = {}
        pct_arrays = []
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

        # Per-seed CSV
        method_df = pd.DataFrame(method_cols)
        method_df.to_csv(RESULTS_DIR / f"{full_name}.csv")
        print(f"  {full_name}.csv ({len(entries)} seeds)")

        # Summary CSV
        pct_stack = np.stack(pct_arrays)
        ordered_cols = dict(sorted(method_cols.items(), key=lambda x: int(x[0].replace("seed", ""))))
        metric_arrays = {m: [] for m in ["ENTROPY", "IMBALANCE", "SHIFT"]}
        for col_name, col_data in ordered_cols.items():
            for m in metric_arrays:
                metric_arrays[m].append(float(col_data[m]))

        summary_rows = []
        for label in labels_by_count:
            idx = all_labels.index(label)
            arr = np.array(pct_stack[:, idx])
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

        label_rows = [r for r in summary_rows if r["metric"] not in {"ENTROPY", "IMBALANCE", "SHIFT"}]
        metric_rows = [r for r in summary_rows if r["metric"] in {"ENTROPY", "IMBALANCE", "SHIFT"}]
        label_rows.sort(key=lambda r: float(r["mean"]), reverse=True)
        ordered_rows = label_rows + metric_rows

        summary_df = pd.DataFrame(ordered_rows).set_index("metric")
        summary_df.to_csv(RESULTS_DIR / f"{full_name}_summary.csv")
        print(f"  {full_name}_summary.csv")

        # Metrics CSV + LaTeX
        metric_rows_only = [r for r in ordered_rows if r["metric"] in {"ENTROPY", "IMBALANCE", "SHIFT"}]
        metrics_table = pd.DataFrame([
            {
                "Metric": r["metric"].capitalize(),
                "Mean": r["mean"],
                "Std": r["std"],
                "Min": r["min"],
                "Max": r["max"],
            }
            for r in metric_rows_only
        ])
        metrics_table.to_csv(RESULTS_DIR / f"{full_name}_metrics.csv", index=False)
        metrics_table.to_latex(
            RESULTS_DIR / f"{full_name}_metrics.tex",
            index=False,
            float_format="%.4f",
            caption=f"Distribution metrics -- {full_name.replace('_', ' ')} ({steps} steps, {len(entries)} seeds)",
            label=f"tab:{full_name}_metrics",
        )
        print(f"  {full_name}_metrics.csv + .tex")

        # Label distribution bar chart
        label_rows_sorted = sorted(label_rows, key=lambda r: float(r["mean"]), reverse=True)
        class_names = [r["metric"] for r in label_rows_sorted]
        means = [float(r["mean"]) for r in label_rows_sorted]
        stds = [float(r["std"]) for r in label_rows_sorted]

        fig, ax = plt.subplots(figsize=(max(14, len(class_names) * 0.35), 7))
        x_pos = np.arange(len(class_names))
        ax.bar(x_pos, means, yerr=stds, align='center', capsize=3, color='steelblue', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, fontsize=7, rotation=90)
        ax.set_ylabel('Mean percentage (%)', fontsize=11)
        ax.set_title(f'Label distribution - {full_name}\n({len(entries)} seeds, {steps} steps)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plot_path = PLOTS_DIR / f"{full_name}_label_distribution.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  {full_name}_label_distribution.png")


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
    PLOTS_DIR.mkdir(exist_ok=True)

    base_counts, base_total = get_distribution(BASE_PARQUET)
    all_labels = sorted(base_counts.index.tolist())
    n_classes = len(all_labels)
    base_pct = pct_vector(base_counts, base_total, all_labels)
    base_entropy = compute_entropy(base_pct, n_classes)
    base_imbalance = imbalance(base_pct, n_classes)

    labels_by_count = sorted(all_labels, key=lambda L: base_counts.get(L, 0), reverse=True)

    # base.csv
    base_data = {}
    for label in labels_by_count:
        c = base_counts.get(label, 0)
        p = (c / base_total * 100.0) if base_total > 0 else 0.0
        base_data[label] = f"{c} ({p:.1f}%)"
    base_data["ENTROPY"] = f"{base_entropy:.3f}"
    base_data["IMBALANCE"] = f"{base_imbalance:.3f}"
    base_data["SHIFT"] = "0.000"
    pd.DataFrame({"base": base_data}).to_csv(RESULTS_DIR / "base.csv")
    print(f"base.csv (entropy={base_entropy:.4f}, imbalance={base_imbalance:.4f})")

    for sim_glob, parse_pattern, method_name in STRATEGIES:
        process_strategy(
            sim_glob, parse_pattern, method_name,
            base_counts, base_total, all_labels, labels_by_count,
            base_pct, n_classes,
        )

    print(f"\nAll results in: {RESULTS_DIR}")
    print(f"All plots in:   {PLOTS_DIR}")


if __name__ == "__main__":
    main()
