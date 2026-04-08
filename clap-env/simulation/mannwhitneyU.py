import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "distribution_results")

COMPARISONS = [
    ("groundtruth_k2_2000",       "groundtruth_k0_2000",       "cold gt k=2 > cold gt k=0"),
    ("groundtruth_k2_2000",       "random_2000",               "cold gt k=2 > random"),
    ("kmeans_hybrid_km57_2000",   "groundtruth_k2_2000",       "kmeans km=57 > cold gt k=2"),
    ("kmeans_hybrid_km100_2000",  "groundtruth_k2_2000",       "kmeans km=100 > cold gt k=2"),
    ("kmeans_hybrid_km150_2000",  "groundtruth_k2_2000",       "kmeans km=150 > cold gt k=2"),
    ("kmeans_hybrid_km100_2000",  "kmeans_hybrid_km57_2000",   "kmeans km=100 > kmeans km=57"),
    ("kmeans_hybrid_km150_2000",  "kmeans_hybrid_km100_2000",  "kmeans km=150 > kmeans km=100"),
]


def load_metric_values(name: str, metric: str) -> list[float]:
    """Load per-seed values for a given metric from a strategy's per-seed CSV."""
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    df = pd.read_csv(path, index_col=0)
    row = df.loc[metric]
    return [float(row[col]) for col in row.index if col.startswith("seed")]


# For entropy: higher is better, so test A > B
# For imbalance: lower is better, so test A < B (i.e. flip alternative)
METRIC_ALTERNATIVE = {
    "ENTROPY":   "greater",
    "IMBALANCE": "less",
}
METRIC_DIRECTION = {
    "ENTROPY":   "A > B (higher entropy is better)",
    "IMBALANCE": "A < B (lower imbalance is better)",
}


def run_test(name_a: str, name_b: str, label: str, metric: str) -> None:
    a = load_metric_values(name_a, metric)
    b = load_metric_values(name_b, metric)

    alternative = METRIC_ALTERNATIVE[metric]
    u_stat, p_value = mannwhitneyu(a, b, alternative=alternative)

    mean_a = np.mean(a)
    mean_b = np.mean(b)
    significant = "YES" if p_value < 0.05 else "NO"

    print(f"  {label}")
    print(f"    {name_a}: mean={mean_a:.4f}  (n={len(a)})")
    print(f"    {name_b}: mean={mean_b:.4f}  (n={len(b)})")
    print(f"    U={u_stat:.1f}, p={p_value:.4f}  -> significant at alpha=0.05: {significant}")
    print()


def main() -> None:
    for metric in ["ENTROPY", "IMBALANCE"]:
        print(f"{'='*55}")
        print(f"  Mann-Whitney U  |  metric: {metric}")
        print(f"  (one-sided: {METRIC_DIRECTION[metric]})")
        print(f"{'='*55}")
        for name_a, name_b, label in COMPARISONS:
            run_test(name_a, name_b, label, metric)


if __name__ == "__main__":
    main()
