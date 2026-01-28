import os

import numpy as np
import pandas as pd

from strategies import (
    get_next_similiar,
    get_next_max_min,
    get_next_meand,
    get_next_groundtruth,
    get_next_random,
)

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
    text_names, text_embeddings = load_text_embeddings()

    base_dataset_name = os.path.basename(PARQUET_PATH).replace(".parquet", "")
    short_name = base_dataset_name.replace("audioset_eval_", "")

    df["is_classified"] = False
    df["clap_labels"] = None
    df["clap_score"] = 0.0

    # THE MEAN EMBEDDING of all recordings first pick
    # all_embeddings = np.stack(df["embedding"].values)
    # all_embeddings = all_embeddings / (
    #     np.linalg.norm(all_embeddings, axis=-1, keepdims=True) + 1e-12
    # )
    # mean_embedding = np.mean(all_embeddings, axis=0)
    # mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)
    # similarities = np.dot(all_embeddings, mean_embedding)
    # first_pick = df.index[np.argmax(similarities)]

    # First pick only for modes that need it (similar, diverse, mean)
    if mode in ("similar", "diverse", "mean"):
        first_pick = df.sample(1).index[0]
        classified_indices = [first_pick]
        first_embedding = df.at[first_pick, "embedding"]
        first_label, first_score = get_clap_label(
            first_embedding, text_embeddings, text_names
        )
        df.at[first_pick, "clap_labels"] = first_label
        df.at[first_pick, "clap_score"] = first_score
        df.at[first_pick, "is_classified"] = True
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
            matching_indices = []
            for idx in df.index:
                if class_name in df.at[idx, "human_labels"]:
                    matching_indices.append(idx)
            available = df.loc[matching_indices]
            available = available[~available["is_classified"]]

            if len(available) < seeds_per_class:
                selected = available.index.tolist()
            else:
                selected = available.sample(seeds_per_class).index.tolist()

            for idx in selected:
                audio_embedding = df.at[idx, "embedding"]
                label, score = get_clap_label(
                    audio_embedding, text_embeddings, text_names
                )
                df.at[idx, "clap_labels"] = label
                df.at[idx, "clap_score"] = score
                df.at[idx, "is_classified"] = True
                classified_indices.append(idx)
                class_counts[class_name] += 1

        # DEBUG: update last_embedding after seeding
        # last_embedding = df.at[classified_indices[-1], "embedding"]
        # print(f"\nAfter seeding ({len(classified_indices)} total), class_counts: {class_counts}")

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
            next_idx = get_next_similiar(df, classified_indices)

        elif mode == "diverse":
            next_idx = get_next_max_min(df)

        elif mode == "mean":
            next_idx = get_next_meand(df, window_size, classified_indices)

        elif mode == "groundtruth":
            next_idx = get_next_groundtruth(
                df, class_counts, classified_indices, window_size
            )

        elif mode == "random":
            next_idx = get_next_random(df)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        if next_idx is None:
            raise ValueError(f"Next pick is None at step {step}")

        audio_embedding = df.at[next_idx, "embedding"]

        # DEBUG: embedding similarity between consecutive picks
        # last_norm = last_embedding / (np.linalg.norm(last_embedding) + 1e-12)
        # audio_norm = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-12)
        # embedding_similarity = np.dot(last_norm, audio_norm)
        # print(f"Step {step}: embedding_sim={embedding_similarity:.3f}")

        label, score = get_clap_label(audio_embedding, text_embeddings, text_names)

        df.at[next_idx, "clap_labels"] = label
        df.at[next_idx, "clap_score"] = score
        df.at[next_idx, "is_classified"] = True
        classified_indices.append(next_idx)

        # DEBUG: update for next iteration
        # last_embedding = audio_embedding

        if mode == "groundtruth":
            human_label = df.at[next_idx, "human_labels"][0]
            class_counts[human_label] += 1

        if step % 100 == 0 or step == steps:
            print(f"Progress: {step}/{steps}")

    print("Completed")

    classified_df = df.loc[
        df["is_classified"],
        ["video_id", "clap_labels", "human_labels"],
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
    import shutil
    import time

    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        print(f"Cleared {OUTPUT_DIR}")

    # TEST MODE
    TEST_MODE = True

    start_time = time.time()

    if TEST_MODE:
        steps = 50
        seeds = [1]
        groundtruth_windows = [5]
        mean_windows = [10]
    else:
        steps = 7 * 225
        seeds = range(1, 31)
        groundtruth_windows = [1, 3, 5, 10]
        mean_windows = [10, 50, 100, 250, 500, 1000]

    for seed in seeds:
        run_simulation(steps=steps, mode="random", seed=seed)

    for window in groundtruth_windows:
        for seed in seeds:
            run_simulation(
                steps=steps, mode="groundtruth", window_size=window, seed=seed
            )

    for seed in seeds:
        run_simulation(steps=steps, mode="similar", seed=seed)

    for seed in seeds:
        run_simulation(steps=steps, mode="diverse", seed=seed)

    for window in mean_windows:
        for seed in seeds:
            run_simulation(steps=steps, mode="mean", window_size=window, seed=seed)

    elapsed = time.time() - start_time

    print("\n")
    print("  +++++++++++++++++++++++++++++++++++++++++++++")
    print("  +                                           +")
    print("  +  PROCESS COMPLETE. MACHINE SPIRIT STABLE. +")
    print("  +  THE OMNISSIAH PROTECTS.                  +")
    print("  +                                           +")
    print("  +++++++++++++++++++++++++++++++++++++++++++++")
    print(f"  ++ OUTPUT :: {OUTPUT_DIR}")
    print(f"  ++ FILES  :: {len(os.listdir(OUTPUT_DIR))}")
    print(f"  ++ TIME   :: {elapsed:.1f} seconds")

    if TEST_MODE:
        # Estimate full run: 390 sims × 1575 steps vs 5 sims × 50 steps
        full_estimate = elapsed * (390 / 5) * (1575 / 50)
        hours = full_estimate / 3600
        print(f"  ++ FULL RUN ESTIMATE :: {hours:.1f} hours")

    print("  ++ END TRANSMISSION ++\n")
