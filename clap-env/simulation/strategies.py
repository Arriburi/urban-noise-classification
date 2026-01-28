import numpy as np


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-12)


def sliding_window_mean(df, classified_indices, window_size):
    if not classified_indices:
        return None

    recent_indices = classified_indices[-window_size:]
    rows = df.loc[recent_indices]

    if rows.empty:
        return None

    embeddings = np.stack(rows["embedding"].values)
    mean_emb = np.mean(embeddings, axis=0)
    return normalize(mean_emb)


def get_next_similiar(df, classified_indices):
    unclassified_mask = df["is_classified"] == False
    unclassified_df = df[unclassified_mask]

    if unclassified_df.empty:
        return None

    rows = df.loc[classified_indices]
    all_classified = np.stack(rows["embedding"].values)
    mean_embedding = np.mean(all_classified, axis=0)
    mean_embedding = normalize(mean_embedding)

    unclassified_embeddings = np.stack(unclassified_df["embedding"].values)
    unclassified_embeddings = normalize(unclassified_embeddings)

    similarities_array = np.dot(unclassified_embeddings, mean_embedding)

    max_similarity_index = np.argmax(similarities_array)

    return unclassified_df.index[max_similarity_index]


def get_next_max_min(df):

    unclassified_mask = df["is_classified"] == False
    unclassified_df = df[unclassified_mask]

    if unclassified_df.empty:
        return None

    classified_mask = df["is_classified"] == True
    classified_df = df[classified_mask]

    if classified_df.empty:
        raise ValueError("No classified samples found for max-min selection")

    U = np.stack(unclassified_df["embedding"].values)
    C = np.stack(classified_df["embedding"].values)

    # normalize embeddings
    U = normalize(U)
    C = normalize(C)

    # dot product of unclassified and classified embeddings
    similarities_matrix = np.dot(U, C.T)

    # find the max similarity for each unclassified item (nearest neighbor)
    max_sims = np.max(similarities_matrix, axis=1)

    # find the item with the lowest max similarity (furthest from any neighbor)
    best_candidate_idx = np.argmin(max_sims)

    return unclassified_df.index[best_candidate_idx]


def get_next_meand(df, window_size, classified_indices):
    unclassified_mask = df["is_classified"] == False
    unclassified_df = df[unclassified_mask]

    if unclassified_df.empty:
        return None

    mean_embedding = sliding_window_mean(df, classified_indices, window_size)
    if mean_embedding is None:
        return None

    unclassified_embeddings = np.stack(unclassified_df["embedding"].values)
    unclassified_embeddings = normalize(unclassified_embeddings)

    similarities = np.dot(unclassified_embeddings, mean_embedding)
    best_candidate_idx = np.argmin(similarities)

    return unclassified_df.index[best_candidate_idx]


def get_next_random(df):
    unclassified = df[~df["is_classified"]]
    if unclassified.empty:
        return None
    return unclassified.sample(1).index[0]


def get_next_groundtruth(df, class_counts, classified_indices, window_size):
    min_count = min(class_counts.values())
    min_classes = [cls for cls, count in class_counts.items() if count == min_count]
    min_class = np.random.choice(min_classes)

    filtered_indices = []
    for idx in classified_indices:
        human_labels = df.at[idx, "human_labels"]
        if min_class in human_labels:
            filtered_indices.append(idx)

    mean_embedding = sliding_window_mean(df, filtered_indices, window_size)

    if mean_embedding is None:
        raise ValueError(
            f"No history found for class '{min_class}' in classified_indices"
        )

    unclassified = df[~df["is_classified"]]
    unclassified_embeddings = np.stack(unclassified["embedding"].values)
    unclassified_embeddings = normalize(unclassified_embeddings)

    similarities = np.dot(unclassified_embeddings, mean_embedding)
    best_idx = np.argmax(similarities)

    return unclassified.index[best_idx]
