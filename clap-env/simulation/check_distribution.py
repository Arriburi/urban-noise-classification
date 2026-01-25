from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


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


def calculate_entropy(counts: pd.Series) -> float:
    """Calculate Shannon entropy for distribution uniformity.
    Higher entropy = more uniform/diverse distribution.
    """
    probabilities = counts / counts.sum()
    return scipy_entropy(probabilities, base=2)


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

    entropy_val = calculate_entropy(counts)
    return counts, total, entropy_val


def main():
    script_dir = Path(__file__).parent

    # BASE DATASET
    BASE_PATH = script_dir / "audioset_eval_top_mixed_no_mixed.parquet"

    base_name = BASE_PATH.stem

    # histograms/<base_name>/
    root_hist_dir = script_dir / "histograms"
    output_dir = root_hist_dir / base_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Histograms will be saved to: {output_dir}")

    print(f"\n{'=' * 70}")
    print("BASE DATASET")
    print(f"{'=' * 70}")
    base_counts, base_total, base_entropy = analyze_file(
        BASE_PATH, "human_labels", output_dir
    )

    # Collect all data for comparison DataFrame
    all_data = {}
    all_entropies = {}
    base_name_short = BASE_PATH.stem
    all_data[base_name_short] = (base_counts, base_total)
    all_entropies[base_name_short] = base_entropy

    # --- SIMULATION FILES ---------------------------------------------------
    files = [
        ("audioset_eval_top_mixed_no_mixed_similar_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_diverse_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_groundtruth_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_meand_n10_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_meand_n50_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_meand_n100_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_meand_n500_1575.parquet", "human_labels"),
        ("audioset_eval_top_mixed_no_mixed_meand_n1000_1575.parquet", "human_labels"),
    ]

    # Add all groundtruth seed files dynamically
    groundtruth_seed_files = list(
        script_dir.glob(
            "audioset_eval_top_mixed_no_mixed_groundtruth_1575_seed*.parquet"
        )
    )
    for seed_file in groundtruth_seed_files:
        files.append((seed_file.name, "human_labels"))

    for file_name, label_col in files:
        sim_path = script_dir / file_name
        if not sim_path.exists():
            print(f"Skipping {file_name} (not found)")
            continue
        counts, total, entropy_val = analyze_file(sim_path, label_col, output_dir)
        all_data[Path(file_name).stem] = (counts, total)
        all_entropies[Path(file_name).stem] = entropy_val

    # Create pivot DataFrame: rows = labels, columns = datasets, values = percentages
    all_labels = set()
    for counts, _ in all_data.values():
        all_labels.update(counts.index)

    # Create shortened column names mapping
    column_mapping = {}
    for dataset_name in all_data.keys():
        # Extract key parts: base name, strategy name, and number
        name_parts = dataset_name.replace("audioset_eval_top_mixed_no_mixed_", "")
        if name_parts == base_name_short:
            column_mapping[dataset_name] = "base"
        else:
            # Further simplify: groundtruth_1575 -> gt, meand_n10_1575 -> meand_n10, etc.
            name_parts = name_parts.replace("groundtruth", "gt")
            # Remove trailing number if it's the same for all (e.g., _1575)
            if "_1575" in name_parts:
                name_parts = name_parts.replace("_1575", "")
            # Keep seed suffix if present (e.g., gt_seed42)
            column_mapping[dataset_name] = name_parts

    comparison_df = pd.DataFrame(
        index=sorted(all_labels), columns=list(all_data.keys())
    )

    for dataset_name, (counts, total) in all_data.items():
        for label in comparison_df.index:
            count = counts.get(label, 0)
            percentage = (count / total * 100) if total > 0 else 0.0
            comparison_df.at[label, dataset_name] = round(percentage, 2)

    comparison_df = comparison_df.fillna(0.0).infer_objects(copy=False)

    # Rename columns using the mapping
    comparison_df = comparison_df.rename(columns=column_mapping)

    # Order rows by highest -> lowest percentage in base dataset column
    base_col = "base"
    if base_col in comparison_df.columns:
        comparison_df = comparison_df.sort_values(by=base_col, ascending=False)

    # Add entropy row at the bottom
    entropy_row = {}
    for dataset_name in all_data.keys():
        short_name = column_mapping[dataset_name]
        entropy_row[short_name] = round(all_entropies[dataset_name], 3)

    # Append entropy as last row
    comparison_df.loc["Entropy"] = entropy_row

    # Save comparison CSV with base dataset name (index=True keeps label names as first column)
    csv_filename = f"{base_name_short}_label_comparison.csv"
    csv_path = output_dir / csv_filename
    comparison_df.to_csv(csv_path, float_format="%.2f")
    print(f"\nLabel comparison saved to: {csv_path}")
    print(f"Shape: {comparison_df.shape} (rows=labels, columns=datasets)")
    print(f"Columns: {', '.join(comparison_df.columns.tolist())}")
    print(f"\nEntropy values (higher = more uniform): {entropy_row}")


if __name__ == "__main__":
    main()
