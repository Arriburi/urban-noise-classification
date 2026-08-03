import os
from glob import glob

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "simulation_outputs")
BASE_DATASET_PARQUET = os.path.join(BASE_DIR, "audioset_eval_mid.parquet")


def label_universe_size(parquet_path: str) -> int:
    df = pd.read_parquet(parquet_path, columns=["human_labels"])
    labels = set()
    for human_labels in df["human_labels"].values:
        if human_labels is not None:
            labels.update(human_labels)
    return len(labels)


def checkpoints(max_step: int = 2000) -> list[int]:
    pts = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000]
    return [p for p in pts if p <= max_step]


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
    max_step = 2000
    cps = checkpoints(max_step)

    strategy_patterns = {
        "random": os.path.join(
            OUTPUT_DIR, "mid_random_2000_seed*.parquet"
        ),
        "kmeans_random_km57": os.path.join(
            OUTPUT_DIR, "mid_kmeans_random_km57_2000_seed*.parquet"
        ),
        "balanced_partition_57": os.path.join(
            OUTPUT_DIR, "mid_balanced_partition_km57_2000_seed*.parquet"
        ),
    }

    # Only include strategies that have matching files — skip missing ones gracefully.
    strategy_files = {
        name: sorted(glob(pattern))
        for name, pattern in strategy_patterns.items()
        if sorted(glob(pattern))
    }
    missing = [name for name in strategy_patterns if name not in strategy_files]
    if missing:
        raise FileNotFoundError(
            "No files matched for required strategy/strategies: " + ", ".join(missing)
        )

    total_labels = label_universe_size(BASE_DATASET_PARQUET)

    means: dict[str, np.ndarray] = {}
    mean_stds: dict[str, float] = {}
    for name, files in strategy_files.items():
        m, s = summarize_strategy(files, total_labels, max_step, cps)
        means[name] = m
        mean_stds[name] = float(np.mean(s))

    table = pd.DataFrame({"step": cps})
    for name in strategy_files:
        table[name] = means[name]

    baseline = "random"
    if baseline not in table:
        raise ValueError("`random` must be present to compute delta columns.")

    for name in strategy_files:
        if name == baseline:
            continue
        table[f"delta_{name}_vs_random"] = table[name] - table[baseline]

    avg_std_row = {"step": "avg_std"}
    for name in strategy_files:
        avg_std_row[name] = mean_stds[name]
    for name in strategy_files:
        if name != baseline:
            avg_std_row[f"delta_{name}_vs_random"] = np.nan
    table = pd.concat([table, pd.DataFrame([avg_std_row])], ignore_index=True)

    console = Console()
    coverage_cols = ["step"] + [name for name in strategy_files.keys()]
    delta_cols = ["step"] + [col for col in table.columns if col.startswith("delta_")]

    def print_rich_table(df: pd.DataFrame, title: str) -> None:
        rich_table = Table(title=title, box=box.SIMPLE_HEAVY, show_lines=False)
        for col in df.columns:
            rich_table.add_column(str(col), justify="right")

        for _, row in df.iterrows():
            cells = []
            for col in df.columns:
                val = row[col]
                if col == "step":
                    cells.append(str(val))
                elif col.startswith("delta_"):
                    num = pd.to_numeric(val, errors="coerce")
                    if pd.isna(num):
                        cells.append("")
                    elif num > 0:
                        cells.append(f"[green]{num:+.2f}[/green]")
                    elif num < 0:
                        cells.append(f"[red]{num:+.2f}[/red]")
                    else:
                        cells.append(f"{num:+.2f}")
                else:
                    num = pd.to_numeric(val, errors="coerce")
                    cells.append("" if pd.isna(num) else f"{num:.2f}")
            rich_table.add_row(*cells)
        console.print(rich_table)

    print_rich_table(table[coverage_cols], "AudioSet Mid Coverage Over Time")
    print_rich_table(table[delta_cols], "Delta vs Random")
    console.print(f"Total labels in base dataset: {total_labels}")
    for name, files in strategy_files.items():
        console.print(f"{name}: {len(files)} files")


if __name__ == "__main__":
    main()
