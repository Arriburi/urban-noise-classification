import os
from glob import glob

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
BASE_DATASET_PARQUET = os.path.join(BASE_DIR, "audioset_eval_mid.parquet")


def label_universe_size(parquet_path: str) -> int:
    df = pd.read_parquet(parquet_path, columns=["human_labels"])
    labels = set()
    for human_labels in df["human_labels"].values:
        if human_labels is not None:
            labels.update(human_labels)
    return len(labels)


def checkpoints(max_step: int = 1000) -> list[int]:
    pts = [0]
    pts.extend(range(10, min(251, max_step + 1), 10))
    pts.extend(range(300, max_step + 1, 100))
    if max_step not in pts:
        pts.append(max_step)
    return sorted(set(pts))


def cumulative_coverage_pct(parquet_path: str, total_labels: int, max_step: int) -> np.ndarray:
    df = pd.read_parquet(parquet_path, columns=["human_labels"])
    seen = set()
    out = np.zeros(max_step + 1, dtype=float)  # index == step; step 0 is 0%
    denom = max(total_labels, 1)
    limit = min(max_step, len(df))
    for step in range(1, limit + 1):
        human_labels = df.iloc[step - 1]["human_labels"]
        if human_labels is not None:
            seen.update(human_labels)
        out[step] = 100.0 * len(seen) / denom
    if limit < max_step:
        out[limit + 1 :] = out[limit]
    return out


def summarize_strategy(
    parquet_paths: list[str], total_labels: int, max_step: int, cps: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    if not parquet_paths:
        raise ValueError("No parquet files provided for strategy.")
    curves = np.stack(
        [cumulative_coverage_pct(p, total_labels, max_step)[cps] for p in parquet_paths]
    )
    mean = curves.mean(axis=0)
    std = curves.std(axis=0, ddof=1) if len(parquet_paths) > 1 else np.zeros_like(mean)
    return mean, std


def main() -> None:
    max_step = 1000
    cps = checkpoints(max_step)

    strategy_patterns = {
        "random_2000": os.path.join(OUTPUT_DIR, "mid_random_2000_seed*.parquet"),
        "diverse_1000": os.path.join(OUTPUT_DIR, "mid_diverse_1000_seed*.parquet"),
    }

    strategy_files = {name: sorted(glob(pattern)) for name, pattern in strategy_patterns.items()}
    missing = [name for name, files in strategy_files.items() if not files]
    if missing:
        raise FileNotFoundError(f"No files matched for: {missing}")

    total_labels = label_universe_size(BASE_DATASET_PARQUET)
    if "random_2000" not in strategy_files:
        raise ValueError("`random_2000` must be present as baseline column.")

    means: dict[str, np.ndarray] = {}
    mean_stds: dict[str, float] = {}
    for name, files in strategy_files.items():
        m, s = summarize_strategy(files, total_labels, max_step, cps)
        means[name] = m
        mean_stds[name] = float(np.mean(s))

    table = pd.DataFrame({"step": cps})
    for name in strategy_files:
        table[name] = means[name]

    random_col = "random_2000"
    for name in strategy_files:
        if name == random_col:
            continue        # examples:

        table[f"delta_vs_random_{name}"] = table[name] - table[random_col]

    avg_std_row = {"step": "avg_std_across_steps"}
    for name in strategy_files:
        avg_std_row[name] = mean_stds[name]
    for name in strategy_files:
        if name == random_col:
            continue
        avg_std_row[f"delta_vs_random_{name}"] = np.nan
    table = pd.concat([table, pd.DataFrame([avg_std_row])], ignore_index=True)

    with pd.option_context("display.max_columns", None, "display.width", 240):
        printable = table.copy()
        for col in printable.columns:
            if col == "step":
                continue
            values = pd.to_numeric(printable[col], errors="coerce")
            if col.startswith("delta_vs_random_"):
                printable[col] = values.map(lambda x: "" if pd.isna(x) else f"{x:+.2f}")
            else:
                printable[col] = values.round(2)
        print(printable.to_string(index=False))

    print(f"Total labels in base dataset: {total_labels}")
    for name, files in strategy_files.items():
        print(f"{name}: {len(files)} files")


if __name__ == "__main__":
    main()
