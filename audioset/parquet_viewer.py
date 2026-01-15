import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Use the no redundant parents file (only leaf labels kept)
df = pd.read_parquet(
    "/home/lucaa/urban-noise-classification/clap-env/simulation/audioset_eval_no_redundant_parents.parquet"
)

total_recordings = len(df)

# Flatten all labels from human_labels (count every label occurrence)
all_labels = []
for labels in df["human_labels"]:
    if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
        all_labels.extend(labels)
    elif labels is not None and not (isinstance(labels, float) and pd.isna(labels)):
        all_labels.append(labels)

label_counts = pd.Series(all_labels).value_counts()
total_labels = len(all_labels)

# Print table: Class name / Label count / Percentage (based on total labels) - Top 20 only
print(f"\n{'Class name':<50} {'Label count':<15} {'Percentage':<10}")
print("-" * 75)
for class_name, count in label_counts.head(20).items():
    pct = (count / total_labels) * 100 if total_labels > 0 else 0.0
    print(f"{class_name:<50} {count:<15} {pct:>6.2f}%")

# Get top 30 for histogram
top_30 = label_counts.head(30)

# Create histogram (one bar per class, height = label count)
plt.figure(figsize=(10, 6))
colors = ["red" if label == "Mixed" else "steelblue" for label in top_30.index]
plt.bar(range(len(top_30)), top_30.values, color=colors)
plt.xlabel("Class", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title(
    f"Recordings: {total_recordings:,}, Labels count: {total_labels:,} (Top 30 shown)",
    fontsize=14,
    fontweight="bold",
)
plt.xticks(range(len(top_30)), top_30.index, rotation=45, ha="right", fontsize=10)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Save histogram
output_dir = "/home/lucaa/urban-noise-classification/clap-env/simulation/histograms"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "label_count_distribution.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.close()

print(f"\nHistogram saved to: {output_path}")