import os

import numpy as np
import pandas as pd

from strategies import (
    get_next_similiar,
    get_next_max_min,
    get_next_meand,
    get_next_groundtruth,
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
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
        print(f"Using random seed: {seed}")

    df = load_parquet()
    text_names, text_embeddings = load_text_embeddings()

    # Extract base dataset name from PARQUET_PATH
    base_dataset_name = os.path.basename(PARQUET_PATH).replace(".parquet", "")

    df["is_classified"] = False
    df["clap_labels"] = None
    df["clap_score"] = 0.0

    print("Picking start point...")
    # Use mean of all embeddings as neutral starting point
    all_embeddings = np.stack(df["embedding"].values)
    mean_embedding = np.mean(all_embeddings, axis=0)
    mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-12)

    # Find the recording closest to the mean
    similarities = np.dot(all_embeddings, mean_embedding)
    first_pick = df.index[np.argmax(similarities)]

    embeddings_history = [df.at[first_pick, "embedding"].copy()]
    first_embedding = df.at[first_pick, "embedding"]
    first_label, first_score = get_clap_label(
        first_embedding, text_embeddings, text_names
    )
    df.at[first_pick, "clap_labels"] = first_label
    df.at[first_pick, "clap_score"] = first_score
    df.at[first_pick, "is_classified"] = True

    last_embedding = first_embedding
    print(
        f"Start Seed: Index {first_pick}, Label: '{first_label}' ({first_score:.2f}) - closest to dataset mean"
    )

    # Initialize class_counts for groundtruth mode
    class_counts = {
        "Human sounds": 0,
        "Animal": 0,
        "Music": 0,
        "Natural sounds": 0,
        "Source-ambiguous sounds": 0,
        "Channel, environment and background": 0,
        "Sounds of things": 0,
    }

    # Groundtruth mode: random initialization for first 50 steps
    if mode == "groundtruth":
        human_label = df.at[first_pick, "human_labels"][0]
        class_counts[human_label] += 1

        for init_step in range(2, 51):
            unclassified = df[~df["is_classified"]]
            random_idx = np.random.choice(unclassified.index)

            audio_embedding = df.at[random_idx, "embedding"]
            label, score = get_clap_label(audio_embedding, text_embeddings, text_names)
            df.at[random_idx, "clap_labels"] = label
            df.at[random_idx, "clap_score"] = score
            df.at[random_idx, "is_classified"] = True

            human_label = df.at[random_idx, "human_labels"][0]
            class_counts[human_label] += 1

        # Debug: print class_counts after random init
        print(f"\nAfter random init (50 steps), class_counts: {class_counts}")
        classes_with_zero = [cls for cls, cnt in class_counts.items() if cnt == 0]
        if classes_with_zero:
            print(f"ERROR: Classes with 0 count: {classes_with_zero}")
            print(f"Seed {seed} failed to hit all classes. Exiting.")
            return
        print("All classes initialized successfully!")

    # Main loop (start from 2 for other modes, from 51 for groundtruth)
    start_step = 51 if mode == "groundtruth" else 2
    for step in range(start_step, steps + 1):
        print(f"Step {step}/{steps}...", end="\r")

        next_embedding = None

        if mode == "similar":
            next_embedding = get_next_similiar(embeddings_history, df)

        elif mode == "diverse":
            next_embedding = get_next_max_min(df)

        elif mode == "mean":
            next_embedding = get_next_meand(df, window_size, embeddings_history)

        elif mode == "groundtruth":
            next_embedding = get_next_groundtruth(df, class_counts)

        else:
            print(f"Unknown mode: {mode}")
            break

        if next_embedding is None:
            print("Next pick is None")
            break

        audio_embedding = df.at[next_embedding, "embedding"]

        # Check similarity between consecutive picks
        last_norm = last_embedding / (np.linalg.norm(last_embedding) + 1e-12)
        audio_norm = audio_embedding / (np.linalg.norm(audio_embedding) + 1e-12)
        embedding_similarity = np.dot(last_norm, audio_norm)

        label, score = get_clap_label(audio_embedding, text_embeddings, text_names)

        df.at[next_embedding, "clap_labels"] = label
        df.at[next_embedding, "clap_score"] = score
        df.at[next_embedding, "is_classified"] = True
        embeddings_history.append(audio_embedding.copy())

        last_embedding = audio_embedding

        # Update class_counts for groundtruth mode
        if mode == "groundtruth":
            human_label = df.at[next_embedding, "human_labels"][0]
            class_counts[human_label] += 1

        print(
            f"Step {step}: Index {next_embedding}, Label: '{label}', Embedding sim: {embedding_similarity:.3f}, CLAP score: {score:.2f}"
        )

    print("\nSimulation finished.")

    classified_df = df.loc[
        df["is_classified"],
        ["video_id", "clap_labels", "human_labels"],
    ].reset_index(drop=True)

    # Build output filename
    if mode == "mean":
        output_filename = f"{base_dataset_name}_{mode}_n{window_size}_{steps}"
    else:
        output_filename = f"{base_dataset_name}_{mode}_{steps}"

    # Add seed suffix if groundtruth mode and seed is provided
    if mode == "groundtruth" and seed is not None:
        output_filename += f"_seed{seed}"

    output_filename += ".parquet"
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    classified_df.to_parquet(output_path, index=False)
    print(f"Saved {len(classified_df)} labeled items to {output_path}")


if __name__ == "__main__":
    steps = 7 * 225  # 1575 steps

    # Test groundtruth with different seeds
    seeds = [42, 789, 1337, 2024, 3141, 9999, 5555, 7777, 8888, 1111]
    for seed in seeds:
        run_simulation(steps=steps, mode="groundtruth", seed=seed)
