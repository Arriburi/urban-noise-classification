import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""
AI generated analysis utilities
"""

# Class rows only (exclude ENTROPY, IMBALANCE, SHIFT)
CLASS_METRICS = {"ENTROPY", "IMBALANCE", "SHIFT"}


def _parse_pct(s: str) -> float:
    if pd.isna(s) or s == "":
        return np.nan
    s = str(s).strip()
    m = re.search(r"\(([\d.]+)%\)", s)
    if m:
        return float(m.group(1))
    if s.endswith("%"):
        return float(s.replace("%", "").strip())
    return float(s)


def plot_method_comparison(results_dir: Path, plots_dir: Path):
    # Read base values for reference lines
    base_path = results_dir / "base.csv"
    base_values = {}
    if base_path.exists():
        base_df = pd.read_csv(base_path, index_col=0)
        if "ENTROPY" in base_df.index and "base" in base_df.columns:
            base_values["ENTROPY"] = float(base_df.loc["ENTROPY", "base"])
        if "IMBALANCE" in base_df.index and "base" in base_df.columns:
            base_values["IMBALANCE"] = float(base_df.loc["IMBALANCE", "base"])

    csv_files = {
        "random": results_dir / "random.csv",
        "similar": results_dir / "similar.csv",
        "diverse": results_dir / "diverse.csv",
        "groundtruth": results_dir / "groundtruth_15.csv",
        "mean": results_dir / "mean_250.csv",
    }

    # Collect data: method -> metric -> list of 30 values (one per seed)
    data = {}
    for method, path in csv_files.items():
        if not path.exists():
            print(f"Warning: {path} not found, skipping {method}")
            continue
        df = pd.read_csv(path, index_col=0)
        data[method] = {}
        for metric in ["ENTROPY", "IMBALANCE", "SHIFT"]:
            if metric not in df.index:
                print(f"Warning: {method} CSV missing {metric}")
                continue
            values = []
            for col in df.columns:
                if col.startswith("seed"):
                    try:
                        values.append(float(df.loc[metric, col]))
                    except (ValueError, KeyError):
                        continue
            if values:
                data[method][metric] = values

    if not data:
        print("No CSV files found, skipping method comparison plots.")
        return

    # Map CSV row names to display titles
    metric_titles = {
        "ENTROPY": "Entropy",
        "IMBALANCE": "Distance to Uniform",
        "SHIFT": "Distance to Base",
    }

    # Create one plot per metric with two subplots
    for metric in ["ENTROPY", "IMBALANCE", "SHIFT"]:
        title = metric_titles.get(metric, metric)
        all_methods = []
        all_value_lists = []
        all_method_labels = []
        main_methods = []
        main_value_lists = []
        main_method_labels = []

        for method in ["random", "similar", "diverse", "groundtruth", "mean"]:
            if method not in data or metric not in data[method]:
                continue
            # Use descriptive labels
            if method == "groundtruth":
                label = "groundtruth_15"
            elif method == "mean":
                label = "mean_250"
            else:
                label = method

            all_methods.append(method)
            all_value_lists.append(data[method][metric])
            all_method_labels.append(label)

            if method != "similar":
                main_methods.append(method)
                main_value_lists.append(data[method][metric])
                main_method_labels.append(label)

        if not all_methods:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Left subplot: all methods (including similar)
        bp1 = ax1.boxplot(
            all_value_lists,
            tick_labels=all_method_labels,
            patch_artist=True,
            whis=(0, 100),
            showfliers=False,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)
        for median in bp1["medians"]:
            median.set_color("black")
        x_positions = range(1, len(all_methods) + 1)
        for x_pos, values in zip(x_positions, all_value_lists):
            mean_val = np.mean(values)
            ax1.plot(x_pos, mean_val, "ko", markersize=4)
            if metric in base_values:
                base_val = base_values[metric]
                ax1.axhline(
                    base_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2.5,
                    zorder=0,
                )
        ax1.set_ylabel(title)
        ax1.set_title(f"{title} (30 seeds)")
        ax1.grid(axis="y", alpha=0.3, linestyle="--")
        ax1.tick_params(axis="x", rotation=45)

        # Right subplot: main methods only (without similar)
        if main_methods:
            bp2 = ax2.boxplot(
                main_value_lists,
                tick_labels=main_method_labels,
                patch_artist=True,
                whis=(0, 100),
                showfliers=False,
            )
            for patch in bp2["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
            for median in bp2["medians"]:
                median.set_color("black")
            x_positions = range(1, len(main_methods) + 1)
            for x_pos, values in zip(x_positions, main_value_lists):
                mean_val = np.mean(values)
                ax2.plot(x_pos, mean_val, "ko", markersize=4)
            if metric in base_values:
                base_val = base_values[metric]
                ax2.axhline(
                    base_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    linewidth=2.5,
                    zorder=0,
                )
            ax2.set_ylabel(title)
            ax2.set_title(f"{title} (30 seeds)")
            ax2.grid(axis="y", alpha=0.3, linestyle="--")
            ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        out_path = plots_dir / f"compare_{metric.lower()}.png"
        plt.savefig(out_path)
        plt.close()
        print(
            f"Created: {out_path.name} ({len(all_methods)} methods total, {len(main_methods)} main)"
        )


def plot_class_distribution_bars(results_dir: Path, plots_dir: Path):
    base_path = results_dir / "base.csv"
    random_summary = results_dir / "random_summary.csv"
    gt_summary = results_dir / "groundtruth_15_summary.csv"
    random_csv = results_dir / "random.csv"
    gt_csv = results_dir / "groundtruth_15.csv"
    if not base_path.exists():
        print("Missing base.csv, skipping class distribution bar plot.")
        return
    if not random_csv.exists() or not gt_csv.exists():
        print(
            "Missing random.csv or groundtruth_15.csv, skipping class distribution bar plot."
        )
        return

    base_df = pd.read_csv(base_path, index_col=0)
    class_labels = [i for i in base_df.index if i not in CLASS_METRICS]
    if not class_labels:
        print("No class rows in base.csv, skipping class distribution bar plot.")
        return

    # Base: one value per class from column "base"
    base_pcts = [_parse_pct(base_df.loc[label, "base"]) for label in class_labels]

    # Random: prefer summary (mean from exact counts), else average parsed % from seed columns
    if random_summary.exists():
        rs_df = pd.read_csv(random_summary, index_col=0)
        random_pcts = [
            float(rs_df.loc[label, "mean"])
            for label in class_labels
            if label in rs_df.index
        ]
        if len(random_pcts) != len(class_labels):
            random_pcts = []
        if not random_pcts:
            rs_df = pd.read_csv(random_csv, index_col=0)
            seed_cols = [c for c in rs_df.columns if c.startswith("seed")]
            random_pcts = [
                np.nanmean([_parse_pct(rs_df.loc[label, c]) for c in seed_cols])
                for label in class_labels
            ]
    else:
        random_df = pd.read_csv(random_csv, index_col=0)
        seed_cols = [c for c in random_df.columns if c.startswith("seed")]
        random_pcts = [
            np.nanmean([_parse_pct(random_df.loc[label, c]) for c in seed_cols])
            for label in class_labels
        ]

    # Groundtruth: same
    if gt_summary.exists():
        gs_df = pd.read_csv(gt_summary, index_col=0)
        gt_pcts = [
            float(gs_df.loc[label, "mean"])
            for label in class_labels
            if label in gs_df.index
        ]
        if len(gt_pcts) != len(class_labels):
            gt_pcts = []
        if not gt_pcts:
            gs_df = pd.read_csv(gt_csv, index_col=0)
            seed_cols = [c for c in gs_df.columns if c.startswith("seed")]
            gt_pcts = [
                np.nanmean([_parse_pct(gs_df.loc[label, c]) for c in seed_cols])
                for label in class_labels
            ]
    else:
        gt_df = pd.read_csv(gt_csv, index_col=0)
        seed_cols = [c for c in gt_df.columns if c.startswith("seed")]
        gt_pcts = [
            np.nanmean([_parse_pct(gt_df.loc[label, c]) for c in seed_cols])
            for label in class_labels
        ]

    # Grouped bar chart: 7 groups, 3 bars per group
    x = np.arange(len(class_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    bars1 = ax.bar(
        x - width, base_pcts, width, label="Base (10k)", color="steelblue", alpha=0.9
    )
    bars2 = ax.bar(
        x,
        random_pcts,
        width,
        label="Random (avg 30 seeds)",
        color="darkorange",
        alpha=0.9,
    )
    bars3 = ax.bar(
        x + width,
        gt_pcts,
        width,
        label="Groundtruth n15 (avg 30 seeds)",
        color="forestgreen",
        alpha=0.9,
    )

    ax.set_ylabel("Average %")
    ax.set_title("Class distribution: Base dataset vs Random vs Groundtruth")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path = plots_dir / "class_distribution_bars.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Created: {out_path.name} (Base, Random, Groundtruth per class)")


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / "distribution_results"
    plots_dir = script_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_method_comparison(results_dir, plots_dir)
    plot_class_distribution_bars(results_dir, plots_dir)
    print("\nDone. PNGs in:", plots_dir)


if __name__ == "__main__":
    main()
