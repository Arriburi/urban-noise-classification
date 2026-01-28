from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy


def get_distribution(file_path: Path) -> tuple[pd.Series, int]:
    df = pd.read_parquet(file_path)
    all_labels = []
    for labels in df["human_labels"]:
        if isinstance(labels, (list, np.ndarray)) and len(labels) > 0:
            all_labels.extend(labels)
    counts = pd.Series(all_labels).value_counts()
    return counts, len(all_labels)


def shorten_name(filename: str) -> str:
    name = filename.replace("top_mixed_no_mixed_", "")
    name = name.replace(".parquet", "")
    name = name.replace("groundtruth", "gt")
    name = name.replace("_1575", "")  # full mode
    name = name.replace("_50", "")    # test mode
    return name


def extract_mode(filename: str) -> str:
    name = shorten_name(filename)
    for mode in ["random", "mean", "gt", "similar", "diverse"]:
        if name.startswith(mode):
            return mode
    return "other"


def build_table(files_data: dict) -> pd.DataFrame:
    all_labels = set()
    for counts, _ in files_data.values():
        all_labels.update(counts.index)

    rows = []
    for label in sorted(all_labels):
        row = {"label": label}
        for name, (counts, total) in files_data.items():
            count = counts.get(label, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            row[name] = f"{count} ({pct:.1f}%)"
        rows.append(row)

    entropy_row = {"label": "ENTROPY"}
    for name, (counts, _) in files_data.items():
        probs = counts / counts.sum()
        entropy_row[name] = f"{scipy_entropy(probs, base=2):.3f}"
    rows.append(entropy_row)

    return pd.DataFrame(rows).set_index("label")


def main():
    script_dir = Path(__file__).parent
    outputs_dir = script_dir / "outputs"
    results_dir = script_dir / "distribution_results"
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
        print(f"Cleared {results_dir}")
    results_dir.mkdir(parents=True, exist_ok=True)

    base_path = script_dir / "audioset_eval_top_mixed_no_mixed.parquet"
    base_data = {"base": get_distribution(base_path)}

    all_files = sorted(outputs_dir.glob("top_mixed_no_mixed_*.parquet"))
    print(f"Found {len(all_files)} files in outputs/\n")

    files_by_mode = {}
    for f in all_files:
        mode = extract_mode(f.name)
        if mode not in files_by_mode:
            files_by_mode[mode] = {}
        short = shorten_name(f.name)
        files_by_mode[mode][short] = get_distribution(f)

    for mode, mode_files in files_by_mode.items():
        data = {**base_data, **mode_files}
        table = build_table(data)
        
        csv_path = results_dir / f"{mode}.csv"
        table.to_csv(csv_path)
        print(f"{mode}: {len(mode_files)} files -> {csv_path.name}")

    representatives = {
        "base": base_data["base"],
        "similar": files_by_mode.get("similar", {}).get("similar"),
        "diverse": files_by_mode.get("diverse", {}).get("diverse"),
        "mean_n250": files_by_mode.get("mean", {}).get("mean_n250"),
        "gt_n5_seed42": files_by_mode.get("gt", {}).get("gt_n5_seed42"),
        "random_seed42": files_by_mode.get("random", {}).get("random_seed42"),
    }
    representatives = {k: v for k, v in representatives.items() if v is not None}

    if len(representatives) > 1:
        summary = build_table(representatives)
        summary_path = results_dir / "summary.csv"
        summary.to_csv(summary_path)
        print(f"\nSummary comparison -> {summary_path.name}")
        print("\n" + summary.to_string())

    print(f"\nResults in: {results_dir}")


if __name__ == "__main__":
    main()
