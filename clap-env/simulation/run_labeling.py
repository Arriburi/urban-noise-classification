import os

import numpy as np
import pandas as pd

from strategies import (
    run_random_mode,
    run_diverse_mode,
    run_kmeans_mode,
    run_balanced_partition_mode,
    run_hdbscan_mode,
    run_groundtruth_mode,
    analyze_groundtruth_hits,
    count_labels,
)


PARQUET_OPTIONS = {
    "audioset_mid": os.path.join(os.path.dirname(__file__), "audioset_eval_mid.parquet"),
    "audioset_mid_fusion": os.path.join(
        os.path.dirname(__file__),
        "parquet_variants",
        "audioset_mid_630k_fusion_HTSAT_tiny.parquet",
    ),
    "urbansound8k": os.path.join(os.path.dirname(__file__), "urbansound8k.parquet"),
    "urbansound": os.path.join(os.path.dirname(__file__), "urbansound.parquet"),
    "esc50": os.path.join(os.path.dirname(__file__), "esc50.parquet"),
}

SCRIPT_DIR = os.path.dirname(__file__)
SIMULATION_OUTPUT_DIR = os.environ.get(
    "SIMULATION_OUTPUT_DIR",
    os.path.join(SCRIPT_DIR, "thesis_seeded", "simulation_outputs"),
)
HIT_RESULTS_DIR = os.environ.get(
    "HIT_RESULTS_DIR",
    os.path.join(SCRIPT_DIR, "thesis_seeded", "hit_results"),
)

DATASET_TOTAL_STEPS = {
    "audioset_mid": 2000,
    "urbansound8k": 1000,
    # Scale from 1000 @ 8.7k samples to ~3600 @ 31.6k samples.
    "urbansound": 3000,
    "esc50": 235,
}

# Groundtruth pre-seeding per dataset (seeded samples do not consume step budget).
DATASET_SEED_PER_LABEL = {
    "audioset_eval_mid": 10,
    "urbansound8k": 10,
    "urbansound": 10,
    "esc50": 3,
}

DEFAULT_TOTAL_STEPS = 300
DEFAULT_SEED_PER_LABEL = 10

# Select dataset(s) here (runs sequentially over this list).
ACTIVE_PARQUETS = ["audioset_mid"]
PARQUET_PATH = PARQUET_OPTIONS[ACTIVE_PARQUETS[0]]

# Enable hit CSV generation for Bayes/no-Bayes ablation analysis.
ENABLE_HIT_ANALYSIS = True


def load_parquet(path: str = PARQUET_PATH):
    return pd.read_parquet(path)


def canonical_dataset_name_for_parquet(parquet_path: str) -> str:
    dataset_name = os.path.basename(parquet_path).replace(".parquet", "")
    if dataset_name.startswith("audioset_mid_"):
        return "audioset_eval_mid"
    return dataset_name


def short_output_name_for_parquet(parquet_path: str) -> str:
    dataset_name = canonical_dataset_name_for_parquet(parquet_path)
    return dataset_name.replace("audioset_eval_", "")


def seed_per_label_for_parquet(parquet_path: str) -> int:
    dataset_name = canonical_dataset_name_for_parquet(parquet_path)
    return DATASET_SEED_PER_LABEL.get(dataset_name, DEFAULT_SEED_PER_LABEL)


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
    parquet_path: str,
    seed: int | None = None,
    window_size: int | None = None,
    dominant_window_size: int | None = None,
    diverse_steps: int = 200,
    groundtruth_steps: int = 1800,
    is_test_mode: bool = False,
    n_clusters: int | None = None,
    kmeans_sub_strategy: str = "diverse",
    hdbscan_min_cluster_size: int | None = None,
    hdbscan_min_samples: int = 1,
    hdbscan_umap_dim: int = 20,
    hdbscan_cluster_order: str = "smallest_first",
    hdbscan_noise_every_cluster_picks: int | None = None,
    enable_bayesian_drop: bool = True,
) -> None:
    if seed is None:
        raise ValueError("seed is required")

    np.random.seed(seed)

    df = load_parquet(parquet_path)

    # Pre-normalize all embeddings once (for cosine similarity)
    all_embeddings = np.stack(df["embedding"].values)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings_norm = all_embeddings / (norms + 1e-12)

    short_name = short_output_name_for_parquet(parquet_path)
    
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
    elif mode == "kmeans":
        if n_clusters is None:
            raise ValueError("n_clusters is required for kmeans mode")
        classified_indices, known_labels, class_counts = run_kmeans_mode(
            df, all_embeddings_norm, steps, seed, n_clusters, kmeans_sub_strategy
        )
        hit_log = None
    elif mode == "balanced_partition":
        if n_clusters is None:
            raise ValueError("n_clusters is required for balanced_partition mode")
        classified_indices, known_labels, class_counts = run_balanced_partition_mode(
            df, all_embeddings_norm, steps, seed, n_clusters
        )
        hit_log = None
    elif mode == "hdbscan":
        classified_indices, known_labels, class_counts = run_hdbscan_mode(
            df, all_embeddings_norm, steps, seed,
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            umap_dim=hdbscan_umap_dim,
            cluster_order=hdbscan_cluster_order,
            noise_every_cluster_picks=hdbscan_noise_every_cluster_picks,
        )
        hit_log = None
    elif mode == "groundtruth":
        # Seed: ensure each label starts with a fixed number of examples.
        # These do NOT consume groundtruth steps.
        seed_per_label = seed_per_label_for_parquet(parquet_path)
        seed_classified_indices, seed_known_labels, seed_class_counts = (
            build_seed_state(df, seed_per_label)
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
                enable_bayesian_drop=enable_bayesian_drop,
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
                enable_bayesian_drop=enable_bayesian_drop,
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
                enable_bayesian_drop=enable_bayesian_drop,
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
        bayes_str = "" if enable_bayesian_drop else "_nb"
        output_filename = (
            f"{short_name}_{mode}_{top_k_str}{bayes_str}_{steps}_seed{seed}.parquet"
        )
    elif mode == "hybrid":
        total_steps = diverse_steps + groundtruth_steps
        output_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_{top_k_str}_seed{seed}.parquet"
    elif mode == "kmeans":
        output_filename = f"{short_name}_kmeans_{kmeans_sub_strategy}_km{n_clusters}_{steps}_seed{seed}.parquet"
    elif mode == "balanced_partition":
        output_filename = f"{short_name}_balanced_partition_km{n_clusters}_{steps}_seed{seed}.parquet"
    elif mode == "hdbscan":
        mc = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else "auto"
        output_filename = (
            f"{short_name}_hdbscan"
            f"_mc{mc}_ms{hdbscan_min_samples}_u{hdbscan_umap_dim}_{hdbscan_cluster_order}"
            f"_ne{hdbscan_noise_every_cluster_picks if hdbscan_noise_every_cluster_picks is not None else 'none'}"
            f"_{steps}_seed{seed}.parquet"
        )
    else:
        output_filename = f"{short_name}_{mode}_{steps}_seed{seed}.parquet"

    output_dir = SIMULATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    if is_test_mode:
        output_filename = f"TEST_{output_filename}"

    output_path = os.path.join(output_dir, output_filename)
    classified_df.to_parquet(output_path, index=False)
    if mode == "kmeans":
        mode_label = f"kmeans_{kmeans_sub_strategy}"
    elif mode == "balanced_partition":
        mode_label = f"balanced_partition_km{n_clusters}"
    elif mode == "hdbscan":
        mc = hdbscan_min_cluster_size if hdbscan_min_cluster_size is not None else "auto"
        mode_label = (
            f"hdbscan_mc{mc}_ms{hdbscan_min_samples}"
            f"_u{hdbscan_umap_dim}_{hdbscan_cluster_order}"
            f"_ne{hdbscan_noise_every_cluster_picks if hdbscan_noise_every_cluster_picks is not None else 'none'}"
        )
    else:
        mode_label = mode.capitalize()
    print(f"[seed {seed}] {mode_label} completed")

    if ENABLE_HIT_ANALYSIS and mode in ("groundtruth", "groundtruth_cold", "hybrid") and hit_log is not None:
        rows = analyze_groundtruth_hits(hit_log, dropped_at_step)

        hit_dir = HIT_RESULTS_DIR
        os.makedirs(hit_dir, exist_ok=True)
        bayes_str = "" if enable_bayesian_drop else "_nb"
        if mode == "hybrid":
            total_steps = diverse_steps + groundtruth_steps
            hit_filename = f"{short_name}_{mode}_{total_steps}_d{diverse_steps}_{top_k_str}_hits_seed{seed}.csv"
        else:
            hit_filename = f"{short_name}_{mode}_{top_k_str}{bayes_str}_{steps}_hits_seed{seed}.csv"
        hit_path = os.path.join(hit_dir, hit_filename)
        pd.DataFrame(rows).to_csv(hit_path, index=False)


def run_task(args):
    (steps, mode, parquet_path, seed, window_size, dominant_window_size,
     top_k, is_test_mode, n_clusters, kmeans_sub_strategy,
     hdbscan_min_cluster_size, hdbscan_min_samples,
     hdbscan_umap_dim, hdbscan_cluster_order,
     hdbscan_noise_every_cluster_picks, enable_bayesian_drop) = args

    import strategies

    original_top_k = strategies.GROUNDTRUTH_TOP_K_DOMINANT
    strategies.GROUNDTRUTH_TOP_K_DOMINANT = top_k
    try:
        run_simulation(
            steps=steps,
            mode=mode,
            parquet_path=parquet_path,
            seed=seed,
            window_size=window_size,
            dominant_window_size=dominant_window_size,
            is_test_mode=is_test_mode,
            n_clusters=n_clusters,
            kmeans_sub_strategy=kmeans_sub_strategy,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            hdbscan_umap_dim=hdbscan_umap_dim,
            hdbscan_cluster_order=hdbscan_cluster_order,
            hdbscan_noise_every_cluster_picks=hdbscan_noise_every_cluster_picks,
            enable_bayesian_drop=enable_bayesian_drop,
        )
    finally:
        strategies.GROUNDTRUTH_TOP_K_DOMINANT = original_top_k


if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    output_dir = SIMULATION_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Toggle this to run a quick smoke test
    TEST_MODE = False

    seeds = range(1, 31)  # 30 seeds
    num_workers = 8

    # AudioSet ablation after DLR calibration:
    # 1) no Bayes, no DLR
    # 2) Bayes, no DLR
    # Existing k=2 outputs can be reused later as Bayes + DLR.
    GROUNDTRUTH_EXPERIMENTS = [
        {"top_k": 0, "enable_bayesian_drop": False},
        {"top_k": 0, "enable_bayesian_drop": True},
    ]

    if TEST_MODE:
        seeds = [1]
        num_workers = 1

    # Use seeded groundtruth for the current AudioSet ablation.
    MODES = ["groundtruth"]
    KMEANS_SUB_STRATEGY = "random"  # "random" or "diverse"
    KMEANS_CLUSTER_SWEEP = [57]

    # HDBSCAN test sweep:
    # (min_cluster_size, min_samples, umap_dim, cluster_order, noise_every_cluster_picks)
    HDBSCAN_CONFIGS = [
        (10, 1, 20, "smallest_first", 3),
        (10, 1, 20, "smallest_first", 5),
    ]

    tasks = []
    for dataset_key in ACTIVE_PARQUETS:
        parquet_path = PARQUET_OPTIONS[dataset_key]
        dataset_steps = DATASET_TOTAL_STEPS.get(dataset_key, DEFAULT_TOTAL_STEPS)
        if TEST_MODE:
            dataset_steps = min(150, dataset_steps)
        for seed in seeds:
            for mode in MODES:
                if mode in ("groundtruth", "groundtruth_cold", "hybrid"):
                    experiment_configs = GROUNDTRUTH_EXPERIMENTS
                else:
                    experiment_configs = [{"top_k": 0, "enable_bayesian_drop": True}]

                for experiment in experiment_configs:
                    top_k = experiment["top_k"]
                    enable_bayesian_drop = experiment["enable_bayesian_drop"]
                    if mode == "hdbscan":
                        for hcfg in HDBSCAN_CONFIGS:
                            tasks.append((
                                dataset_steps, mode, parquet_path, seed, None, None, top_k,
                                TEST_MODE, n_clusters, KMEANS_SUB_STRATEGY,
                                hcfg[0], hcfg[1], hcfg[2], hcfg[3], hcfg[4],
                                enable_bayesian_drop,
                            ))
                    elif mode in ("kmeans", "balanced_partition"):
                        for n_clusters in KMEANS_CLUSTER_SWEEP:
                            tasks.append((
                                dataset_steps, mode, parquet_path, seed, None, None, top_k,
                                TEST_MODE, n_clusters, KMEANS_SUB_STRATEGY,
                                None, 1, 20, "smallest_first", None,
                                enable_bayesian_drop,
                            ))
                    else:
                        tasks.append((
                            dataset_steps, mode, parquet_path, seed, None, None, top_k,
                            TEST_MODE, max(1, len(load_parquet(parquet_path)) // 50), KMEANS_SUB_STRATEGY,
                            None, 1, 20, "smallest_first", None,
                            enable_bayesian_drop,
                        ))

    print(f"Running {len(tasks)} simulations with {num_workers} workers...")
    print(f"  Datasets: {ACTIVE_PARQUETS}")
    print(f"  Modes: {MODES}")
    print(f"  K-means sweep: {KMEANS_CLUSTER_SWEEP}")
    print(f"  Groundtruth experiments: {GROUNDTRUTH_EXPERIMENTS}")
    print(f"  Seeds: {len(list(seeds))}")
    print("  Steps per dataset:")
    for dataset_key in ACTIVE_PARQUETS:
        steps = DATASET_TOTAL_STEPS.get(dataset_key, DEFAULT_TOTAL_STEPS)
        if TEST_MODE:
            steps = min(150, steps)
        print(f"    - {dataset_key}: {steps}")
    print()

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
