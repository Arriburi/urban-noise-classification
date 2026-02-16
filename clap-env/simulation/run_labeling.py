import os

import numpy as np
import pandas as pd

from collections import Counter

from strategies import (
    get_next_similiar,
    get_next_max_min,
    get_next_meand,
    get_next_groundtruth,
    get_next_random,
    update_mean,
)


def analyze_groundtruth_hits(hit_log):
    targets = sorted(set(t for t, _ in hit_log))
    rows = []

    total_hits = 0
    total_all = 0

    for cls in targets:
        entries = [(t, a) for t, a in hit_log if t == cls]
        hits = sum(1 for t, a in entries if t == a)
        misses = len(entries) - hits
        total = len(entries)
        hit_rate = hits / total if total > 0 else 0.0

        total_hits += hits
        total_all += total

        miss_classes = Counter(a for t, a in entries if t != a)
        miss_detail = "; ".join(f"{c}: {n}" for c, n in miss_classes.most_common())

        rows.append(
            {
                "target_class": cls,
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": round(hit_rate, 4),
                "miss_detail": miss_detail if miss_detail else "-",
            }
        )

    overall_rate = total_hits / total_all if total_all > 0 else 0.0
    rows.append(
        {
            "target_class": "OVERALL",
            "hits": total_hits,
            "misses": total_all - total_hits,
            "total": total_all,
            "hit_rate": round(overall_rate, 4),
            "miss_detail": "-",
        }
    )

    return rows


PARQUET_PATH = os.path.join(
    os.path.dirname(__file__), "audioset_eval_top_mixed_no_mixed.parquet"
)
TEXT_EMBEDDING_PATH = os.path.join(
    os.path.dirname(__file__), "..", "clap_text_embeddings.npz"
)


def load_parquet(path=PARQUET_PATH):
    return pd.read_parquet(path)


def load_text_embeddings(path=TEXT_EMBEDDING_PATH):
    data = np.load(path, allow_pickle=True)
    names = data["names"]
    embeddings = data["embeddings"]

    # not normalized in text_embedding.py
    norm = np.linalg.norm(embeddings, axis=-1, keepdims=True)
    embeddings = embeddings / (norm + 1e-12)

    return names, embeddings


def get_clap_label(audio_embedding, text_embeddings, class_names):
    audio_norm = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-12)

    similarities = np.dot(text_embeddings, audio_norm)
    best_idx = np.argmax(similarities)

    closest_label = class_names[best_idx]
    best_score = similarities[best_idx]

    return closest_label, best_score


def run_simulation(steps, mode, window_size=None, seed=None):
    if seed is None:
        raise ValueError("seed is required")

    if mode in ("groundtruth", "mean") and window_size is None:
        raise ValueError(f"window_size is required for mode '{mode}'")

    np.random.seed(seed)
    print(f"Running mode: {mode}")

    df = load_parquet()
    # text_names, text_embeddings = load_text_embeddings()

    # Pre-normalize all embeddings once (for cosine similarity)
    all_embeddings = np.stack(df["embedding"].values)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    all_embeddings_norm = all_embeddings / (norms + 1e-12)

    # Boolean array for tracking classified status
    n_samples = len(df)
    is_classified = np.zeros(n_samples, dtype=bool)

    base_dataset_name = os.path.basename(PARQUET_PATH).replace(".parquet", "")
    short_name = base_dataset_name.replace("audioset_eval_", "")

    running_mean = None

    # THE MEAN EMBEDDING of all recordings first pick
    # all_embeddings = np.stack(df["embedding"].values)
    # all_embeddings = all_embeddings / (
    #     np.linalg.norm(all_embeddings, axis=-1, keepdims=True) + 1e-12
    # )
    # mean_embedding = np.mean(all_embeddings, axis=0)
    # mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)
    # similarities = np.dot(all_embeddings, mean_embedding)
    # first_pick = df.index[np.argmax(similarities)]

    # First pick only for modes that need it (similar, diverse, mean). Use position (iloc) consistently.
    if mode in ("similar", "diverse", "mean"):
        first_pick_index = df.sample(1).index[0]
        first_pick = df.index.get_loc(first_pick_index)
        classified_indices = [first_pick]
        first_embedding = all_embeddings[first_pick]
        if mode in ("similar", "mean"):
            running_mean = first_embedding.copy()
        is_classified[first_pick] = True
    else:
        # groundtruth and random don't need first_pick
        classified_indices = []

    # Initialize class_counts for groundtruth mode
    class_names_list = [
        "Human sounds",
        "Animal",
        "Music",
        "Natural sounds",
        "Source-ambiguous sounds",
        "Channel, environment and background",
        "Sounds of things",
    ]
    class_counts = {name: 0 for name in class_names_list}
    seeds_per_class = 5

    if mode == "groundtruth":
        for class_name in class_names_list:
            # Find samples with this class label that aren't classified yet. Store positions (iloc).
            mask = df["human_labels"].apply(lambda x: class_name in x)
            available = df[mask & ~pd.Series(is_classified, index=df.index)]

            if len(available) < seeds_per_class:
                selected_index = available.index.tolist()
            else:
                selected_index = available.sample(seeds_per_class).index.tolist()
            selected_pos = [df.index.get_loc(i) for i in selected_index]

            for pos in selected_pos:
                is_classified[pos] = True
                classified_indices.append(pos)
                class_counts[class_name] += 1

    # Track hit/miss for groundtruth mode
    hit_log = []

    if mode == "groundtruth":
        start_step = len(class_names_list) * seeds_per_class + 1
    elif mode in ("similar", "diverse", "mean"):
        start_step = 2
    else:
        # random starts from step 1
        start_step = 1

    for step in range(start_step, steps + 1):
        next_idx = None

        if mode == "similar":
            next_idx = get_next_similiar(
                all_embeddings_norm, is_classified, running_mean
            )

        elif mode == "diverse":
            next_idx = get_next_max_min(all_embeddings_norm, is_classified)

        elif mode == "mean":
            next_idx = get_next_meand(all_embeddings_norm, is_classified, running_mean)

        elif mode == "groundtruth":
            next_idx, target_class = get_next_groundtruth(
                all_embeddings,
                all_embeddings_norm,
                is_classified,
                df,
                class_counts,
                classified_indices,
                window_size,
            )

        elif mode == "random":
            next_idx = get_next_random(df, is_classified)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if next_idx is None:
            raise ValueError(f"Next pick is None at step {step}")

        audio_embedding = all_embeddings[next_idx]

        is_classified[next_idx] = True
        classified_indices.append(next_idx)

        if mode == "similar":
            running_mean = update_mean(
                running_mean, audio_embedding, len(classified_indices)
            )
        if mode == "mean":
            if len(classified_indices) <= window_size:
                oldest_embedding = None
                total_count = len(classified_indices)
            else:
                oldest_idx = classified_indices[-(window_size + 1)]
                oldest_embedding = all_embeddings[oldest_idx]
                total_count = window_size
            running_mean = update_mean(
                running_mean, audio_embedding, total_count, oldest_embedding
            )

        if mode == "groundtruth":
            human_label = df.iloc[next_idx]["human_labels"][0]
            class_counts[human_label] += 1
            hit_log.append((target_class, human_label))

    # --- Groundtruth hit/miss analysis ---
    if mode == "groundtruth" and hit_log:
        hit_summary = analyze_groundtruth_hits(hit_log)
        hit_df = pd.DataFrame(hit_summary)

        hit_filename = f"{short_name}_{mode}_n{window_size}_hits_{steps}_seed{seed}.csv"
        HIT_DIR = os.path.join(os.path.dirname(__file__), "hit_results")
        os.makedirs(HIT_DIR, exist_ok=True)
        hit_path = os.path.join(HIT_DIR, hit_filename)
        hit_df.to_csv(hit_path, index=False)
        print(f"\nHit/miss analysis saved to {hit_path}")
        print(hit_df.to_string(index=False))
        print()

    classified_df = df.iloc[classified_indices][
        ["video_id", "human_labels"]
    ].reset_index(drop=True)

    if mode in ("mean", "groundtruth"):
        output_filename = f"{short_name}_{mode}_n{window_size}_{steps}"
    else:
        output_filename = f"{short_name}_{mode}_{steps}"

    output_filename += f"_seed{seed}"

    output_filename += ".parquet"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    classified_df.to_parquet(output_path, index=False)
    print(f"Saved {len(classified_df)} labeled items to {output_path}")


if __name__ == "__main__":
    import time
    from multiprocessing import Pool

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    TEST_MODE = False

    if TEST_MODE:
        steps = 200
        seeds = [1, 2, 3]
        groundtruth_windows = [3, 5]
        mean_windows = [10, 50]
        num_workers = 4
    else:
        steps = 7 * 225
        seeds = range(1, 31)
        groundtruth_windows = [15]  # Windows go from 1 to 50 with 10/15 interval
        mean_windows = [10, 50, 100, 250, 500, 1000]
        num_workers = 10  # got 12

    tasks = []
    for seed in seeds:
        # tasks.append((steps, "random", None, seed))
        # tasks.append((steps, "similar", None, seed))
        # tasks.append((steps, "diverse", None, seed))
        for window in groundtruth_windows:
            tasks.append((steps, "groundtruth", window, seed))
        # for window in mean_windows:
        #     tasks.append((steps, "mean", window, seed))

    print(f"Running {len(tasks)} simulations with {num_workers} workers...")

    def run_task(args):
        steps, mode, window_size, seed = args
        run_simulation(steps=steps, mode=mode, window_size=window_size, seed=seed)

    start_time = time.time()
    total_tasks = len(tasks)

    with Pool(processes=num_workers) as pool:
        completed = 0
        for _ in pool.imap(run_task, tasks):
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
        print()  # Newline after progress bar

    elapsed = time.time() - start_time

    print("\n")
    print("  +++++++++++++++++++++++++++++++++++++++++++++")
    print(f"  ++ OUTPUT :: {OUTPUT_DIR}")
    print(f"  ++ FILES  :: {len(tasks)}")
    print(f"  ++ TIME   :: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print("  ++ END TRANSMISSION ++\n")
