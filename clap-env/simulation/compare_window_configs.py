import os, re
from collections import Counter
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import entropy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
STEPS, MODE = 2000, "groundtruth"
PATTERN = re.compile(rf"^mid_{MODE}_(n\w+)_(d\w+)_{STEPS}_seed(\d+)\.parquet$")


def final_entropy(path: str) -> float:
    labels = [l for hs in pd.read_parquet(path)["human_labels"].values if hs is not None for l in hs]
    if not labels:
        return 0.0
    c = Counter(labels)
    p = np.array(list(c.values()), dtype=float)
    p /= p.sum()
    return float(entropy(p, base=2))


def main() -> None:
    grouped: dict[tuple[str, str], list[float]] = {}
    for fname in os.listdir(OUTPUT_DIR):
        m = PATTERN.match(fname)
        if not m:
            continue
        key = (m.group(1), m.group(2))
        grouped.setdefault(key, []).append(final_entropy(os.path.join(OUTPUT_DIR, fname)))
    if not grouped:
        raise FileNotFoundError(f"No files like {PATTERN.pattern}")

    rows = []
    for (win, dom), vals in grouped.items():
        arr = np.array(vals, dtype=float)
        rows.append(
            {
                "configuration": f"target_win={win[1:]}, dom_win={dom[1:]}",
                "n_seeds": len(arr),
                "mean_entropy": float(arr.mean()),
                "std_entropy": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min_entropy": float(arr.min()),
                "max_entropy": float(arr.max()),
                "_vals": arr,
            }
        )
    rows.sort(key=lambda r: r["mean_entropy"], reverse=True)

    os.makedirs(PLOTS_DIR, exist_ok=True)
    csv_path = os.path.join(PLOTS_DIR, f"window_comparison_{MODE}_{STEPS}steps.csv")
    pd.DataFrame([{k: v for k, v in r.items() if k != "_vals"} for r in rows]).to_csv(csv_path, index=False)

    labels = [r["configuration"] for r in rows]
    means = [r["mean_entropy"] for r in rows]
    stds = [r["std_entropy"] for r in rows]
    dists = [r["_vals"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(labels))
    ax1.bar(x, means, yerr=stds, capsize=4, color="steelblue", alpha=0.8)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Final Entropy (bits)"); ax1.grid(axis="y", alpha=0.3)
    bp = ax2.boxplot(dists, labels=labels, patch_artist=True)
    for b in bp["boxes"]:
        b.set_facecolor("lightblue")
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Final Entropy (bits)"); ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"window_comparison_{MODE}_{STEPS}steps.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

    best = rows[0]
    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Best: {best['configuration']} -> {best['mean_entropy']:.4f} ± {best['std_entropy']:.4f}")


if __name__ == "__main__":
    main()
