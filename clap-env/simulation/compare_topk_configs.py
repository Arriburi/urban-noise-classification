import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Configuration
STEPS = 1500
MODE = "groundtruth"
TOP_K_VALUES = [0, 1, 2, 3, 5, 10]
SEEDS = list(range(1, 11))


def compute_entropy_at_step(labels_up_to_step):
    if not labels_up_to_step:
        return 0.0
    
    counter = Counter(labels_up_to_step)
    counts = np.array(list(counter.values()))
    probabilities = counts / counts.sum()
    
    return entropy(probabilities, base=2)


def load_and_compute_final_entropy(filepath):
    df = pd.read_parquet(filepath)
    
    all_labels = []
    for human_labels in df["human_labels"].values:
        if human_labels is not None:
            all_labels.extend(human_labels)
    
    return compute_entropy_at_step(all_labels)


def main():
    print(f"Loading outputs from: {OUTPUT_DIR}")
    print(f"Mode: {MODE}, Steps: {STEPS}\n")
    print(f"Top-k values (exact): {TOP_K_VALUES}")
    print(f"Seeds (exact): {SEEDS}\n")
    
    # Group files by top-k value using exact expected filenames only
    by_topk = {k: {} for k in TOP_K_VALUES}  # top_k -> {seed: entropy}
    missing_files = []

    for top_k in TOP_K_VALUES:
        for seed in SEEDS:
            fname = f"mid_{MODE}_k{top_k}_{STEPS}_seed{seed}.parquet"
            filepath = os.path.join(OUTPUT_DIR, fname)
            if not os.path.exists(filepath):
                missing_files.append(fname)
                continue
            by_topk[top_k][seed] = load_and_compute_final_entropy(filepath)
    
    available_topk = {k: v for k, v in by_topk.items() if v}
    if not available_topk:
        raise FileNotFoundError(f"No matching files found for {MODE} with {STEPS} steps")
    if missing_files:
        print(f"Warning: missing {len(missing_files)} expected files.")
        for name in missing_files[:10]:
            print(f"  - {name}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
        print()
    
    print(f"Found {len(available_topk)} top-k configurations with available files:\n")
    
    # Compute statistics for each top-k
    results = []
    for top_k in sorted(available_topk.keys()):
        entropies = list(available_topk[top_k].values())
        seeds = sorted(available_topk[top_k].keys())
        
        mean_entropy = np.mean(entropies)
        std_entropy = np.std(entropies, ddof=1)
        min_entropy = np.min(entropies)
        max_entropy = np.max(entropies)
        
        # Identify outliers using IQR method (same as boxplot)
        q1 = np.percentile(entropies, 25)
        q3 = np.percentile(entropies, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_info = []
        for seed in seeds:
            ent = by_topk[top_k][seed]
            if ent < lower_bound or ent > upper_bound:
                outlier_info.append(f"seed {seed}: {ent:.4f}")
        
        print(f"top-k = {top_k}: mean entropy: {mean_entropy:.4f} ± {std_entropy:.4f}")
        if outlier_info:
            print(f"  Outliers: {', '.join(outlier_info)}")
        
        results.append({
            'top_k': top_k,
            'n_seeds': len(seeds),
            'mean_entropy': mean_entropy,
            'std_entropy': std_entropy,
            'min_entropy': min_entropy,
            'max_entropy': max_entropy,
            'entropies': entropies,
            'outliers': outlier_info,
        })
    
    # Sort by top-k value
    results.sort(key=lambda x: x['top_k'])
    
    # Save summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            'top_k': r['top_k'],
            'n_seeds': r['n_seeds'],
            'mean_entropy': f"{r['mean_entropy']:.4f}",
            'std_entropy': f"{r['std_entropy']:.4f}",
            'min_entropy': f"{r['min_entropy']:.4f}",
            'max_entropy': f"{r['max_entropy']:.4f}",
        })
    
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(BASE_DIR, "plots", f"topk_comparison_{MODE}_{STEPS}steps.csv")
    os.makedirs(os.path.join(BASE_DIR, "plots"), exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"Summary saved to: {csv_path}\n")
    
    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Line plot with error bars
    top_ks = [r['top_k'] for r in results]
    means = [r['mean_entropy'] for r in results]
    stds = [r['std_entropy'] for r in results]
    
    ax1.errorbar(top_ks, means, yerr=stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax1.set_xlabel('Top-k (number of dominant classes)', fontsize=12)
    ax1.set_ylabel('Final Entropy (bits)', fontsize=12)
    ax1.set_title(f'Mean Final Entropy by Top-k\n({STEPS} steps, error bars = std)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(top_ks)
    
    # Plot 2: Box plots showing distribution
    entropy_distributions = [r['entropies'] for r in results]
    labels = [f"k={k}" for k in top_ks]
    bp = ax2.boxplot(entropy_distributions, labels=labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax2.set_xlabel('Top-k', fontsize=12)
    ax2.set_ylabel('Final Entropy (bits)', fontsize=12)
    ax2.set_title(f'Entropy Distribution by Top-k\n({STEPS} steps, {results[0]["n_seeds"]} seeds)', fontsize=13)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(BASE_DIR, "plots", f"topk_comparison_{MODE}_{STEPS}steps.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")
    
    # Print best configuration
    best = max(results, key=lambda x: x['mean_entropy'])
    print(f"\nBest top-k: {best['top_k']}")
    print(f"  Mean entropy: {best['mean_entropy']:.4f} ± {best['std_entropy']:.4f}")


if __name__ == "__main__":
    main()
