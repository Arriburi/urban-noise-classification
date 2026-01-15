import os

import numpy as np
import pandas as pd

from strategies import get_next_similiar, get_next_max_min, get_next_cluster, get_next_meand

PARQUET_PATH = "/home/lucaa/urban-noise-classification/clap-env/simulation/audioset_eval_top_non_mixed.parquet"
TEXT_EMBEDDING_PATH = "/home/lucaa/urban-noise-classification/clap-env/clap_text_embeddings.npz"

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

def run_simulation(steps, mode, window_size=None):
    df = load_parquet()
    text_names, text_embeddings = load_text_embeddings()

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
    first_label, first_score = get_clap_label(first_embedding, text_embeddings, text_names)
    df.at[first_pick, "clap_labels"] = first_label
    df.at[first_pick, "clap_score"] = first_score
    df.at[first_pick, "is_classified"] = True

    last_embedding = first_embedding
    print(f"Start Seed: Index {first_pick}, Label: '{first_label}' ({first_score:.2f}) - closest to dataset mean")

    # sim Loop (start from 2 since step 1 was the initial pick)
    for step in range(2, steps + 1):
        print(f"Step {step}/{steps}...", end="\r")

        next_embedding = None

        if mode == "similar":
            next_embedding = get_next_similiar(last_embedding, df)

        elif mode == "diverse":
            next_embedding = get_next_max_min(df)
        
        elif mode == "mean":
            next_embedding = get_next_meand(df, window_size, embeddings_history)

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

        print(f"Step {step}: Index {next_embedding}, Label: '{label}', Embedding sim: {embedding_similarity:.3f}, CLAP score: {score:.2f}")
            
    print("\nSimulation finished.")

    classified_df = df.loc[
        df["is_classified"],
        ["video_id", "clap_labels", "human_labels"],
    ].reset_index(drop=True)

    if mode == "mean":
        output_filename = f"simulation2_meand_n{window_size}_{steps}.parquet"
    else:
        output_filename = f"simulation2_{mode}_{steps}.parquet"
    output_path = os.path.join(os.path.dirname(__file__), output_filename)
    classified_df.to_parquet(output_path, index=False)
    print(f"Saved {len(classified_df)} labeled items to {output_path}")

if __name__ == "__main__":
    steps = 7 * 225  # 1575 steps
    run_simulation(steps=steps, mode="similar")
    run_simulation(steps=steps, mode="diverse")
    run_simulation(steps=steps, mode="mean", window_size=10)
    run_simulation(steps=steps, mode="mean", window_size=50)
    run_simulation(steps=steps, mode="mean", window_size=100)
    run_simulation(steps=steps, mode="mean", window_size=500)
