"""Compare final entropy across top-k groundtruth configurations."""

from __future__ import annotations

import os
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy

matplotlib.use("Agg")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get(
    "TOPK_OUTPUT_DIR",
    os.path.join(BASE_DIR, "thesis_seeded", "simulation_outputs"),
)
PLOTS_DIR = os.environ.get(
    "TOPK_PLOTS_DIR",
    os.path.join(BASE_DIR, "thesis_seeded", "plots"),
)

STEPS = int(os.environ.get("TOPK_STEPS", "2000"))
MODE = os.environ.get("TOPK_MODE", "groundtruth")
TOP_K_VALUES = [int(value) for value in os.environ.get("TOPK_VALUES", "1,2,3,4,5").split(",")]
SEEDS = [int(value) for value in os.environ.get("TOPK_SEEDS", ",".join(str(i) for i in range(1, 31))).split(",")]


def compute_final_entropy(filepath: str) -> float:
    df = pd.read_parquet(filepath, columns=["human_labels"])
    counts: Counter[str] = Counter()
    for human_labels in df["human_labels"].values:
        if human_labels is not None:
            counts.update(human_labels)
    if not counts:
        return 0.0
    freq = np.array(list(counts.values()), dtype=float)
    probs = freq / freq.sum()
    return float(entropy(probs, base=2))


def main() -> None:
    print(f"Loading outputs from: {OUTPUT_DIR}")
    print(f"Mode: {MODE}, Steps: {STEPS}\n")
    print(f"Top-k values (exact): {TOP_K_VALUES}")
    print(f"Seeds (exact): {SEEDS}\n")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    missing_files: list[str] = []
    summary_rows: list[dict[str, float | int]] = []
    distributions: list[list[float]] = []
    available_top_ks: list[int] = []

    for top_k in TOP_K_VALUES:
        entropies: list[float] = []
        for seed in SEEDS:
            path = os.path.join(OUTPUT_DIR, f"mid_{MODE}_k{top_k}_{STEPS}_seed{seed}.parquet")
            if not os.path.exists(path):
                missing_files.append(os.path.basename(path))
                continue
            entropies.append(compute_final_entropy(path))

        if not entropies:
            continue

        arr = np.array(entropies, dtype=float)
        summary_rows.append({
            "top_k": top_k,
            "n_seeds": int(len(arr)),
            "mean_entropy": float(arr.mean()),
            "std_entropy": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "min_entropy": float(arr.min()),
            "max_entropy": float(arr.max()),
        })
        distributions.append(entropies)
        available_top_ks.append(top_k)

    if missing_files:
        print(f"Warning: missing {len(missing_files)} expected files.")
        for filename in missing_files[:10]:
            print(f"  - {filename}")
        if len(missing_files) > 10:
            print("  ...")
        print()

    if not summary_rows:
        raise FileNotFoundError("No matching top-k simulation outputs found.")

    summary_df = pd.DataFrame(summary_rows).sort_values("top_k").reset_index(drop=True)

    print(f"Found {len(summary_df)} top-k configurations with available files:\n")
    for row in summary_df.itertuples(index=False):
        print(
            f"top-k = {row.top_k}: mean entropy = "
            f"{row.mean_entropy:.4f} +/- {row.std_entropy:.4f}"
        )

    output_stem = f"topk_comparison_{MODE}_{STEPS}steps"
    csv_path = os.path.join(PLOTS_DIR, f"{output_stem}.csv")
    summary_df.to_csv(csv_path, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].errorbar(
        summary_df["top_k"],
        summary_df["mean_entropy"],
        yerr=summary_df["std_entropy"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#1f5aa6",
    )
    axes[0].set_xlabel("Top-k (stevilo dominantnih razredov)")
    axes[0].set_ylabel("Koncna entropija (biti)")
    axes[0].set_title(
        f"Povprecna koncna entropija glede na top-k\n({STEPS} korakov, napake = std)"
    )
    axes[0].grid(True, linestyle="--", alpha=0.3)

    bp = axes[1].boxplot(
        distributions,
        tick_labels=[f"k={value}" for value in available_top_ks],
        patch_artist=True,
    )
    for box in bp["boxes"]:
        box.set(facecolor="#9ec7d8", edgecolor="black", linewidth=1.0)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)
    for median in bp["medians"]:
        median.set(color="#ff7f0e", linewidth=1.0)

    axes[1].set_xlabel("Top-k")
    axes[1].set_ylabel("Koncna entropija (biti)")
    axes[1].set_title(
        f"Porazdelitev entropije glede na top-k\n({STEPS} korakov, {len(SEEDS)} seeds)"
    )
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(PLOTS_DIR, f"{output_stem}.png")
    plt.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close()

    best_row = summary_df.loc[summary_df["mean_entropy"].idxmax()]
    print(f"\nSummary saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}\n")
    print(f"Best top-k: {int(best_row['top_k'])}")
    print(
        f"  Mean entropy: {best_row['mean_entropy']:.4f} +/- "
        f"{best_row['std_entropy']:.4f}"
    )


if __name__ == "__main__":
    main()
