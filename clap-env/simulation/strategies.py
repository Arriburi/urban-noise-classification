import numpy as np


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-12)


def sliding_window_mean(all_embeddings, classified_indices, window_size):
    if not classified_indices:
        return None

    recent_indices = classified_indices[-window_size:]
    embeddings = all_embeddings[recent_indices]
    mean_emb = np.mean(embeddings, axis=0)
    return normalize(mean_emb)

    # welford


def update_mean(mean_embedding, new_embedding, total_count, oldest_embedding=None):
    if oldest_embedding is None:
        return mean_embedding + (new_embedding - mean_embedding) / total_count
    else:
        return mean_embedding + (new_embedding - oldest_embedding) / total_count


def get_next_similiar(all_embeddings_norm, is_classified, running_mean):
    if np.all(is_classified):
        return None

    mean_embedding = normalize(running_mean)
    similarities = all_embeddings_norm @ mean_embedding
    similarities[is_classified] = -np.inf

    return np.argmax(similarities)


def get_next_max_min(all_embeddings_norm, is_classified):
    if np.all(is_classified):
        return None

    if not np.any(is_classified):
        raise ValueError("No classified samples found for max-min selection")

    C = all_embeddings_norm[is_classified]

    # Compute similarities for ALL samples against classified
    similarities_matrix = all_embeddings_norm @ C.T

    # Find max similarity to any classified sample (nearest neighbor)
    max_sims = np.max(similarities_matrix, axis=1)

    # Mask out already classified
    max_sims[is_classified] = np.inf

    # Find sample with lowest max similarity (furthest from any neighbor)
    return np.argmin(max_sims)


def get_next_meand(all_embeddings_norm, is_classified, running_mean):
    if np.all(is_classified):
        return None

    mean_embedding = normalize(running_mean)
    similarities = all_embeddings_norm @ mean_embedding
    similarities[is_classified] = np.inf  # Use +inf for argmin

    return np.argmin(similarities)


def get_next_random(df, is_classified):
    # Use df.sample() to maintain random state compatibility. Return position (iloc) for consistency.
    unclassified = df[~is_classified]
    if unclassified.empty:
        return None
    sampled_index = unclassified.sample(1).index[0]
    return df.index.get_loc(sampled_index)


def get_next_groundtruth(
    all_embeddings,
    all_embeddings_norm,
    is_classified,
    df,
    class_counts,
    classified_indices,
    window_size,
):
    min_count = min(class_counts.values())
    min_classes = [cls for cls, count in class_counts.items() if count == min_count]
    min_class = np.random.choice(min_classes)

    filtered_indices = []
    for idx in classified_indices:
        human_labels = df.iloc[idx]["human_labels"]
        if min_class in human_labels:
            filtered_indices.append(idx)

    mean_embedding = sliding_window_mean(all_embeddings, filtered_indices, window_size)

    if mean_embedding is None:
        raise ValueError(
            f"No history found for class '{min_class}' in classified_indices"
        )

    similarities = all_embeddings_norm @ mean_embedding
    similarities[is_classified] = -np.inf

    return np.argmax(similarities), min_class
