import pandas as pd
import numpy as np
from pathlib import Path

def analyze_file(file_path, label_column):
    df = pd.read_parquet(file_path)
    file_name = Path(file_path).name
    
    print(f"\n{'='*70}")
    print(f"File: {file_name}")
    print(f"Label column: {label_column}")
    print(f"{'='*70}")
    print(f"Total samples: {len(df)}")
    
    # human labels
    if label_column == "human_labels":
        all_labels = []
        for labels in df[label_column]:
            if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
                all_labels.extend(labels)
        counts = pd.Series(all_labels).value_counts()
        total = len(all_labels)
    else:
        # clap_labels strings
        classes = df[label_column].dropna()
        counts = classes.value_counts()
        total = len(classes)
    
    print(f"Total label occurrences: {total}")
    print(f"Unique classes: {len(counts)}\n")
    print(f"{'Class':<50} {'Label count':<15} {'Percentage':<10}")
    print("-" * 75)
    
    for class_name, count in counts.head(30).items():
        pct = (count / total) * 100
        print(f"{class_name:<50} {count:<15} {pct:>6.2f}%")
    
    if len(counts) > 30:
        print(f"\n... and {len(counts) - 30} more classes")
    
    # entropy calc
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(counts))  # Maximum possible entropy
    print(f"\nEntropy: {entropy:.4f} (max possible: {max_entropy:.4f}, {entropy/max_entropy*100:.1f}% of max)")
    
    return counts, total, entropy


def main():
    script_dir = Path(__file__).parent
    

    base_path = "/home/lucaa/urban-noise-classification/audioset/audioset_eval.parquet"
    print(f"\n{'='*70}")
    print("BASE DATASET (20k full dataset)")
    print(f"{'='*70}")
    _, _, base_entropy = analyze_file(base_path, "human_labels")
    
    files = [
        ("simulation_meand_n10_8000.parquet", "human_labels"),
        ("simulation_meand_n50_8000.parquet", "human_labels"),
        ("simulation_meand_n100_8000.parquet", "human_labels"),
        ("simulation_meand_n500_8000.parquet", "human_labels"),
        ("simulation_meand_n1000_8000.parquet", "human_labels"),
        ("simulation_results_diverse_8000.parquet", "human_labels"),
        ("simulation_similar_8000.parquet", "human_labels")
    ]
    
    results = []
    for file, label_col in files:
        _, _, entropy = analyze_file(script_dir / file, label_col)
        results.append((Path(file).stem, entropy))
    
    results.sort(key=lambda x: x[1], reverse=True)
    print(f"\n{'='*70}")
    print("Uniformity Ranking (by Entropy - higher = more uniform):")
    print(f"{'='*70}")
    print(f"BASE (20k dataset):{' '*25} Entropy: {base_entropy:.4f}")
    print("-" * 70)
    for i, (name, ent) in enumerate(results, 1):
        diff = ent - base_entropy
        print(f"{i}. {name:<40} Entropy: {ent:.4f} ({diff:+.4f} vs base)")


if __name__ == "__main__":
    main()
