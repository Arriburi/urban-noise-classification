import os

import numpy as np
import pandas as pd

from strategies import (
    run_random_mode,
    run_diverse_mode,
    run_groundtruth_mode,
    analyze_groundtruth_hits,
)


PARQUET_PATH = os.path.join(
    os.path.dirname(__file__),
    "audioset_eval_mid.parquet",
)


def load_parquet(path=PARQUET_PATH):
    return pd.read_parquet(path)


def run_simulation(
    steps: int,
    mode: str,
    seed: int | None = None,
    window_size: int = 15,
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
        classified_indices, known_labels, class_counts, hit_log = run_groundtruth_mode(
            df,
            all_embeddings,
            all_embeddings_norm,
            steps,
            seed,
            window_size,
        )
    elif mode == "hybrid":
        # Phase 1: diverse for coverage (seed labels for groundtruth)
        classified_indices, known_labels, class_counts = run_diverse_mode(
            df, all_embeddings_norm, diverse_steps, seed
        )
        # Phase 2: groundtruth for balancing (continues from diverse state)
        classified_indices, known_labels, class_counts, hit_log = run_groundtruth_mode(
            df,
            all_embeddings,
            all_embeddings_norm,
            groundtruth_steps,
            seed,
            window_size,
            initial_classified_indices=classified_indices,
            initial_known_labels=known_labels,
            initial_class_counts=class_counts,
        )
        steps = diverse_steps + groundtruth_steps
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # --- Save output ---
    classified_df = df.iloc[classified_indices][
        ["video_id", "human_labels"]
    ].reset_index(drop=True)

    if mode == "groundtruth":
        output_filename = (
            f"{short_name}_{mode}_n{window_size}_{steps}_seed{seed}.parquet"
        )
    elif mode == "hybrid":
        total_steps = diverse_steps + groundtruth_steps
        output_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_n{window_size}_seed{seed}.parquet"
    else:
        output_filename = f"{short_name}_{mode}_{steps}_seed{seed}.parquet"

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, output_filename)
    classified_df.to_parquet(output_path, index=False)
    print(f"[seed {seed}] Saved {len(classified_df)} labeled items to {output_path}")
    print(f"[seed {seed}] Final known classes: {len(known_labels)}")

    if mode in ("groundtruth", "hybrid") and hit_log is not None:
        rows = analyze_groundtruth_hits(hit_log)

        # Save full per-class stats for this seed to CSV for offline analysis
        hit_dir = os.path.join(os.path.dirname(__file__), "hit_results")
        os.makedirs(hit_dir, exist_ok=True)
        if mode == "hybrid":
            total_steps = diverse_steps + groundtruth_steps
            hit_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_n{window_size}_hits_seed{seed}.csv"
        else:
            hit_filename = (
                f"{short_name}_{mode}_n{window_size}_{steps}_hits_seed{seed}.csv"
            )
        hit_path = os.path.join(hit_dir, hit_filename)
        pd.DataFrame(rows).to_csv(hit_path, index=False)
        print(f"[seed {seed}] Groundtruth hit stats saved to {hit_path}")


if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Toggle this to run a quick smoke test
    TEST_MODE = False

    # New plan: fix groundtruth steps at 1000 and sweep diverse.
    # This creates a controlled experiment where only the diverse phase length varies:
    #   - d200  + gt1000 = 1200 total
    #   - d300  + gt1000 = 1300 total
    #   - d400  + gt1000 = 1400 total
    #   - d500  + gt1000 = 1500 total
    #   - d600  + gt1000 = 1600 total
    #   - d750  + gt1000 = 1750 total
    #   - d1000 + gt1000 = 2000 total
    #   - d1250 + gt1000 = 2250 total
    #   - d1500 + gt1000 = 2500 total
    #   - d1800 + gt1000 = 2800 total
    #   - d2000 + gt1000 = 3000 total
    GT_STEPS = 1000
    DIVERSE_OPTIONS = [200, 300, 400, 500, 600, 750]

    seeds = range(1, 6)  # 5 seeds per diverse setting by default
    num_workers = 10

    if TEST_MODE:
        # Single tiny config for quick local smoke test
        DIVERSE_OPTIONS = [50]  # 50 diverse + 150 groundtruth
        GT_STEPS = 150
        seeds = [1]
        num_workers = 1

    tasks = []
    for diverse_steps in DIVERSE_OPTIONS:
        total_steps = diverse_steps + GT_STEPS
        groundtruth_steps = GT_STEPS
        for seed in seeds:
            tasks.append(
                (total_steps, "hybrid", seed, diverse_steps, groundtruth_steps)
            )

    print(f"Running {len(tasks)} simulations with {num_workers} workers...")
    print("  Mode: hybrid")
    print("  Configs (total_steps = diverse + groundtruth):")
    for diverse_steps in DIVERSE_OPTIONS:
        total_steps = diverse_steps + GT_STEPS
        print(f"    - {total_steps} = d{diverse_steps} + gt{GT_STEPS}")
    print(f"  Seeds: {list(seeds)}")
    print()

    def run_task(args):
        steps, mode, seed, diverse_steps, groundtruth_steps = args
        run_simulation(
            steps=steps,
            mode=mode,
            seed=seed,
            diverse_steps=diverse_steps,
            groundtruth_steps=groundtruth_steps,
        )

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
