import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_histogram(counts: pd.Series, file_path: Path, output_dir: Path):
    file_name = file_path.name
    top = counts.head(30)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top)), top.values, color="steelblue")
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title(
        f"Label Count Distribution: {Path(file_name).stem}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(
        range(len(top)),
        top.index,
        rotation=45,
        ha="right",
        fontsize=10,
    )
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{Path(file_name).stem}_histogram.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to: {output_path}")


def analyze_file(file_path: Path, label_column: str, output_dir: Path | None = None):
    df = pd.read_parquet(file_path)
    file_name = file_path.name

    print(f"\n{'=' * 70}")
    print(f"File: {file_name}")
    print(f"Label column: {label_column}")
    print(f"{'=' * 70}")
    print(f"Total samples: {len(df)}")

    # human_labels = lists/arrays; others (e.g. clap_labels) = strings
    if label_column == "human_labels":
        all_labels = []
        for labels in df[label_column]:
            if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
                all_labels.extend(labels)
        counts = pd.Series(all_labels).value_counts()
        total = len(all_labels)
    else:
        classes = df[label_column].dropna()
        counts = classes.value_counts()
        total = len(classes)

    print(f"Total label occurrences: {total}")
    print(f"Unique classes: {len(counts)}\n")
    print(f"{'Class':<50} {'Label count':<15} {'Percentage':<10}")
    print("-" * 75)

    for class_name, count in counts.head(30).items():
        pct = (count / total) * 100 if total > 0 else 0.0
        print(f"{class_name:<50} {count:<15} {pct:>6.2f}%")

    if len(counts) > 30:
        print(f"\n... and {len(counts) - 30} more classes")

    if output_dir is not None:
        create_histogram(counts, file_path, output_dir)

    return counts, total


def main():
    script_dir = Path(__file__).parent

    # BASE DATASET 
    BASE_PATH = script_dir / "audioset_eval_top_non_mixed.parquet"

    base_name = BASE_PATH.stem

    # histograms/<base_name>/
    root_hist_dir = script_dir / "histograms"
    output_dir = root_hist_dir / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Histograms will be saved to: {output_dir}")

    print(f"\n{'=' * 70}")
    print("BASE DATASET")
    print(f"{'=' * 70}")
    base_counts, base_total = analyze_file(BASE_PATH, "human_labels", output_dir)
    
    # Collect all data for comparison DataFrame
    all_data = {}
    base_name_short = BASE_PATH.stem
    all_data[base_name_short] = (base_counts, base_total)

    # --- SIMULATION FILES ---------------------------------------------------
    files = [
        ("simulation2_diverse_1575.parquet", "human_labels"),
        ("simulation2_meand_n10_1575.parquet", "human_labels"),
        ("simulation2_meand_n50_1575.parquet", "human_labels"),
        ("simulation2_meand_n100_1575.parquet", "human_labels"),
        ("simulation2_meand_n500_1575.parquet", "human_labels"),
        ("simulation2_similar_1575.parquet", "human_labels"),
    ]

    for file_name, label_col in files:
        sim_path = script_dir / file_name
        counts, total = analyze_file(sim_path, label_col, output_dir)
        all_data[Path(file_name).stem] = (counts, total)
    
    # Create pivot DataFrame: rows = labels, columns = datasets, values = percentages
    all_labels = set()
    for counts, _ in all_data.values():
        all_labels.update(counts.index)
    
    comparison_df = pd.DataFrame(index=sorted(all_labels), columns=list(all_data.keys()))
    
    for dataset_name, (counts, total) in all_data.items():
        for label in comparison_df.index:
            count = counts.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0.0
            comparison_df.at[label, dataset_name] = round(percentage, 2)
    
    comparison_df = comparison_df.fillna(0.0)
    
    # Order rows by highest -> lowest percentage in base dataset column
    base_col = base_name_short  # e.g. 'audioset_eval_top_non_mixed'
    if base_col in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=base_col, ascending=False)
    
    # Save comparison CSV
    csv_path = output_dir / "label_comparison.csv"
    comparison_df.to_csv(csv_path, float_format='%.2f')
    print(f"\nLabel comparison saved to: {csv_path}")
    print(f"Shape: {comparison_df.shape} (rows=labels, columns=datasets)")


if __name__ == "__main__":
    main()

