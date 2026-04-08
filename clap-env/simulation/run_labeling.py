import os

import numpy as np
import pandas as pd

from strategies import (
    run_random_mode,
    run_diverse_mode,
    run_groundtruth_mode,
    analyze_groundtruth_hits,
    count_labels,
)


PARQUET_PATH = os.path.join(
    os.path.dirname(__file__),
    "audioset_eval_mid.parquet",
)

# Disable hit CSV generation during speed-focused sweeps.
ENABLE_HIT_ANALYSIS = False


def load_parquet(path=PARQUET_PATH):
    return pd.read_parquet(path)


def build_seed_state(df: pd.DataFrame, seed_per_label: int):
    """
    Build seed state for groundtruth (not cold).
    Every label has `seed_per_label` recordings.

    These seeded labels do NOT count towards the groundtruth step budget.
    """
    n_samples = len(df)
    label_to_indices: dict[str, list[int]] = {}

    # Map each label to all row indices where it appears
    for row_idx, human_labels in enumerate(df["human_labels"].values):
        if human_labels is None:
            continue
        for label in human_labels:
            label_to_indices.setdefault(label, []).append(row_idx)

    is_classified = np.zeros(n_samples, dtype=bool)
    classified_indices: list[int] = []
    known_labels: set[str] = set()
    class_counts: dict[str, int] = {}

    # For each label, pick exactly `seed_per_label` unique recordings.
    # If a label does not have enough recordings, fail fast.
    for label, indices in label_to_indices.items():
        if len(indices) < seed_per_label:
            raise ValueError(
                f"Label '{label}' has only {len(indices)} recordings, "
                f"but seed_per_label={seed_per_label} is required."
            )

        chosen = np.random.choice(indices, size=seed_per_label, replace=False)
        for idx in chosen:
            if not is_classified[idx]:
                is_classified[idx] = True
                classified_indices.append(idx)
                human_labels = df.iloc[idx]["human_labels"]
                count_labels(human_labels, known_labels, class_counts)

    return classified_indices, known_labels, class_counts


def run_simulation(
    steps: int,
    mode: str,
    seed: int | None = None,
    window_size: int | None = None,
    dominant_window_size: int | None = None,
    diverse_steps: int = 200,
    groundtruth_steps: int = 1800,
) -> None:
    if seed is None:
        raise ValueError("seed is required")

    np.random.seed(seed)

    df = load_parquet()

    # Pre-normalize all embeddings once (for cosine similarity)
    all_embeddings = np.stack(df["embedding"].values)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings_norm = all_embeddings / (norms + 1e-12)

    base_dataset_name = os.path.basename(PARQUET_PATH).replace(".parquet", "")
    short_name = base_dataset_name.replace("audioset_eval_", "")
    
    # Create config suffix for filename
    win_str = f"n{window_size}" if window_size is not None else "nNone"
    dom_str = f"d{dominant_window_size}" if dominant_window_size is not None else "dNone"
    
    # Import to get current top-k value
    from strategies import GROUNDTRUTH_TOP_K_DOMINANT
    top_k_str = f"k{GROUNDTRUTH_TOP_K_DOMINANT}"

    if mode == "random":
        classified_indices, known_labels, class_counts = run_random_mode(
            df, all_embeddings_norm, steps, seed
        )
        hit_log = None
    elif mode == "diverse":
        classified_indices, known_labels, class_counts = run_diverse_mode(
            df, all_embeddings_norm, steps, seed
        )
        hit_log = None
    elif mode == "groundtruth":
        # Seed: ensure each label starts with a fixed number of examples.
        # These do NOT consume groundtruth steps.
        SEED_PER_LABEL = 10
        seed_classified_indices, seed_known_labels, seed_class_counts = (
            build_seed_state(df, SEED_PER_LABEL)
        )

        classified_indices, known_labels, class_counts, hit_log, dropped_at_step = (
            run_groundtruth_mode(
                df,
                all_embeddings,
                all_embeddings_norm,
                steps,
                seed,
                window_size,
                dominant_window_size=dominant_window_size,
                initial_classified_indices=seed_classified_indices,
                initial_known_labels=seed_known_labels,
                initial_class_counts=seed_class_counts,
            )
        )
    elif mode == "groundtruth_cold":
        # No pre-seeding: groundtruth starts from a single random sample (warm start only)
        classified_indices, known_labels, class_counts, hit_log, dropped_at_step = (
            run_groundtruth_mode(
                df,
                all_embeddings,
                all_embeddings_norm,
                steps,
                seed,
                window_size,
                dominant_window_size=dominant_window_size,
            )
        )
    elif mode == "hybrid":
        # Phase 1: diverse for coverage (seed labels for groundtruth)
        classified_indices, known_labels, class_counts = run_diverse_mode(
            df, all_embeddings_norm, diverse_steps, seed
        )
        # Phase 2: groundtruth for balancing (continues from diverse state)
        classified_indices, known_labels, class_counts, hit_log, dropped_at_step = (
            run_groundtruth_mode(
                df,
                all_embeddings,
                all_embeddings_norm,
                groundtruth_steps,
                seed,
                window_size,
                dominant_window_size=dominant_window_size,
                initial_classified_indices=classified_indices,
                initial_known_labels=known_labels,
                initial_class_counts=class_counts,
            )
        )
        steps = diverse_steps + groundtruth_steps
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Save output ---
    classified_df = df.iloc[classified_indices][
        ["video_id", "human_labels"]
    ].reset_index(drop=True)

    if mode in ("groundtruth", "groundtruth_cold"):
        output_filename = (
            f"{short_name}_{mode}_{top_k_str}_{steps}_seed{seed}.parquet"
        )
    elif mode == "hybrid":
        total_steps = diverse_steps + groundtruth_steps
        output_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_{top_k_str}_seed{seed}.parquet"
    else:
        output_filename = f"{short_name}_{mode}_{steps}_seed{seed}.parquet"

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    classified_df.to_parquet(output_path, index=False)
    print(f"[seed {seed}] {mode.capitalize()} completed")

    if ENABLE_HIT_ANALYSIS and mode in ("groundtruth", "groundtruth_cold", "hybrid") and hit_log is not None:
        rows = analyze_groundtruth_hits(hit_log, dropped_at_step)

        hit_dir = os.path.join(os.path.dirname(__file__), "hit_results")
        os.makedirs(hit_dir, exist_ok=True)
        if mode == "hybrid":
            total_steps = diverse_steps + groundtruth_steps
            hit_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_{top_k_str}_hits_seed{seed}.csv"
        else:
            hit_filename = f"{short_name}_{mode}_{top_k_str}_{steps}_hits_seed{seed}.csv"
        hit_path = os.path.join(hit_dir, hit_filename)
        pd.DataFrame(rows).to_csv(hit_path, index=False)


if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Toggle this to run a quick smoke test
    TEST_MODE = False

    TOTAL_STEPS = 1500
    seeds = range(1, 11)  # 10 seeds
    num_workers = 10
    TOP_K_VALUES = [0, 1, 2, 3, 5, 10]

    if TEST_MODE:
        TOTAL_STEPS = 100
        seeds = [1]
        num_workers = 1

    MODES = ["groundtruth"]

    tasks = []
    for seed in seeds:
        for top_k in TOP_K_VALUES:
            for mode in MODES:
                tasks.append((TOTAL_STEPS, mode, seed, None, None, top_k))

    print(f"Running {len(tasks)} simulations with {num_workers} workers...")
    print(f"  Modes: {MODES}")
    print(f"  Top-k values: {TOP_K_VALUES}")
    print(f"  Seeds: {len(list(seeds))}")
    print(f"  Steps per seed: {TOTAL_STEPS}")
    print()

    def run_task(args):
        steps, mode, seed, window_size, dominant_window_size, top_k = args

        import strategies
        original_top_k = strategies.GROUNDTRUTH_TOP_K_DOMINANT
        strategies.GROUNDTRUTH_TOP_K_DOMINANT = top_k
        try:
            run_simulation(
                steps=steps,
                mode=mode,
                seed=seed,
                window_size=window_size,
                dominant_window_size=dominant_window_size,
            )
        finally:
            strategies.GROUNDTRUTH_TOP_K_DOMINANT = original_top_k

    start_time = time.time()
    total_tasks = len(tasks)

    with Pool(processes=num_workers) as pool:
        completed = 0
        for _ in pool.imap_unordered(run_task, tasks):
            completed += 1
            progress = completed / total_tasks
            bar_width = 40
            filled = int(bar_width * progress)
            bar = "#" * filled + "-" * (bar_width - filled)
            print(
                f"\r[{bar}] {progress * 100:5.1f}% ({completed}/{total_tasks} simulations)",
                end="",
                flush=True,
            )
        print()

    elapsed = time.time() - start_time

    print("\n")
    print("  +++++++++++++++++++++++++++++++++++++++++++++")
    print(f"  ++ OUTPUT :: {output_dir}")
    print(f"  ++ FILES  :: {len(tasks)}")
    print(f"  ++ TIME   :: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("  ++ END TRANSMISSION ++\n")
