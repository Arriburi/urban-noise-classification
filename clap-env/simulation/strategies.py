import numpy as np
import pandas as pd
from scipy.stats import beta as _beta_dist  # type: ignore[attr-defined]


GROUNDTRUTH_TOP_K_DOMINANT = 3


def normalize(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-12)


def sliding_window_mean(all_embeddings, classified_indices, window_size):
    if not classified_indices:
        return None

    if window_size is None:
        recent_indices = classified_indices
    else:
        recent_indices = classified_indices[-window_size:]
    embeddings = all_embeddings[recent_indices]
    mean_emb = np.mean(embeddings, axis=0)
    return normalize(mean_emb)


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


def should_drop_label_bayesian(
    hits: int,
    attempts: int,
    min_useful_hit_rate: float = 0.05,
    confidence: float = 0.95,
) -> bool:
    """
    Decide whether to drop a label based on its hit/attempt history.

    - min_useful_hit_rate: lowest hit rate you still care about (e.g. 0.05 = 5%)
    - confidence: how sure you want to be that the true hit rate is below that
                  before dropping (e.g. 0.95 = 95%)

    Returns True if the label looks bad enough to stop targeting.
    """
    if attempts <= 0:
        return False  # no evidence, never drop

    # Prior: Beta(1, 1) = uniform, weak and agnostic
    alpha = 1.0 + hits
    beta_param = 1.0 + (attempts - hits)

    # Upper credible bound u such that P(p <= u | data) = confidence
    upper_bound = float(_beta_dist.ppf(confidence, alpha, beta_param))

    # Drop if even the upper plausible hit rate is below what we consider useful
    return upper_bound < min_useful_hit_rate


def get_next_max_min(all_embeddings_norm, is_classified):
    if np.all(is_classified):
        return None

    if not np.any(is_classified):
        raise ValueError("No classified samples found for max-min selection")

    C = all_embeddings_norm[is_classified]

    similarities_matrix = all_embeddings_norm @ C.T
    max_sims = np.max(similarities_matrix, axis=1)
    max_sims[is_classified] = np.inf
    return np.argmin(max_sims)


def get_next_meand(all_embeddings_norm, is_classified, running_mean):
    if np.all(is_classified):
        return None

    mean_embedding = normalize(running_mean)
    similarities = all_embeddings_norm @ mean_embedding
    similarities[is_classified] = np.inf

    return np.argmin(similarities)


def get_next_random(df, is_classified):
    unclassified = df[~is_classified]
    if unclassified.empty:
        return None
    sampled_index = unclassified.sample(1).index[0]
    return df.index.get_loc(sampled_index)


def get_top_k_dominant_labels(class_counts: dict[str, int], target_class: str, top_k: int):
    items = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    filtered = [(label, count) for label, count in items if label != target_class]
    if not filtered or top_k <= 0:
        return []
    top = filtered[:top_k]
    total = sum(count for _, count in top)
    if total <= 0:
        return []
    return [(label, count / total) for label, count in top]


def get_next_groundtruth(
    all_embeddings,
    all_embeddings_norm,
    is_classified,
    human_labels_by_idx,
    class_counts,
    classified_indices,
    label_centroid_cache,
    window_size,
    dominant_window_size=None,
    dropped_classes=None,
):
    if dropped_classes is None:
        dropped_classes = set()

    # Choose among classes that have NOT been dropped
    active_items = [
        (cls, count) for cls, count in class_counts.items() if cls not in dropped_classes
    ]
    if not active_items:
        # All known classes have been dropped; fall back to using all classes
        active_items = list(class_counts.items())

    counts_only = [count for _, count in active_items]
    min_count = min(counts_only)
    min_classes = [cls for cls, count in active_items if count == min_count]
    min_class = np.random.choice(min_classes)

    # Fast path: no windows -> use incremental cached centroids.
    if window_size is None:
        state = label_centroid_cache.get(min_class)
        mean_embedding = state["mean"] if state is not None else None
    else:
        filtered_indices = []
        for idx in classified_indices:
            human_labels = human_labels_by_idx[idx]
            if min_class in human_labels:
                filtered_indices.append(idx)
        mean_embedding = sliding_window_mean(all_embeddings, filtered_indices, window_size)

    if mean_embedding is None:
        raise ValueError(
            f"No classified recordings found for class '{min_class}'. "
            f"Diverse cold-start phase should have discovered this class."
        )

    # Attraction to target class centroid.
    scores = all_embeddings_norm @ mean_embedding

    # DLR: subtract weighted similarities to top-k dominant class centroids.
    dominant = get_top_k_dominant_labels(
        class_counts=class_counts,
        target_class=min_class,
        top_k=GROUNDTRUTH_TOP_K_DOMINANT,
    )
    for label, weight in dominant:
        if dominant_window_size is None:
            dom_state = label_centroid_cache.get(label)
            dom_centroid = dom_state["mean"] if dom_state is not None else None
        else:
            label_indices = []
            for idx in classified_indices:
                human_labels = human_labels_by_idx[idx]
                if label in human_labels:
                    label_indices.append(idx)
            dom_centroid = sliding_window_mean(all_embeddings, label_indices, dominant_window_size)
        if dom_centroid is None:
            continue
        penalty = all_embeddings_norm @ dom_centroid
        scores -= weight * penalty

    scores[is_classified] = -np.inf
    return int(np.argmax(scores)), min_class


def count_labels(human_labels, known_labels, class_counts):
    for label in human_labels:
        if label not in known_labels:
            known_labels.add(label)
            class_counts[label] = 0
        class_counts[label] += 1


def run_random_mode(df, all_embeddings_norm, steps, seed):
    n_samples = len(df)
    is_classified = np.zeros(n_samples, dtype=bool)
    known_labels = set()
    class_counts = {}
    classified_indices = []
    human_labels_by_idx = df["human_labels"].values

    for step in range(1, steps + 1):
        next_idx = get_next_random(df, is_classified)
        if next_idx is None:
            raise ValueError(f"Random returned None at step {step}")

        is_classified[next_idx] = True
        classified_indices.append(next_idx)

        human_labels = human_labels_by_idx[next_idx]
        count_labels(human_labels, known_labels, class_counts)

    return classified_indices, known_labels, class_counts


def run_diverse_mode(df, all_embeddings_norm, steps, seed):
    n_samples = len(df)
    is_classified = np.zeros(n_samples, dtype=bool)
    known_labels = set()
    class_counts = {}
    classified_indices = []
    human_labels_by_idx = df["human_labels"].values

    first_pick_index = df.sample(1).index[0]
    first_pick = df.index.get_loc(first_pick_index)
    is_classified[first_pick] = True
    classified_indices.append(first_pick)

    human_labels = human_labels_by_idx[first_pick]
    count_labels(human_labels, known_labels, class_counts)

    for step in range(2, steps + 1):
        next_idx = get_next_max_min(all_embeddings_norm, is_classified)
        if next_idx is None:
            raise ValueError(f"Diverse returned None at step {step}")

        is_classified[next_idx] = True
        classified_indices.append(next_idx)

        human_labels = human_labels_by_idx[next_idx]
        count_labels(human_labels, known_labels, class_counts)

    return classified_indices, known_labels, class_counts


def run_groundtruth_mode(
    df,
    all_embeddings,
    all_embeddings_norm,
    steps,
    seed,
    window_size,
    dominant_window_size=None,
    initial_classified_indices=None,
    initial_known_labels=None,
    initial_class_counts=None,
):
    """
    Groundtruth-guided selection over the actual label space
    (no coarse top-layer grouping).

    Optional initial state (e.g. from diverse phase) for hybrid runs:
    - initial_classified_indices: list of row indices already classified
    - initial_known_labels: set of labels discovered so far
    - initial_class_counts: dict label -> count
    """
    n_samples = len(df)
    is_classified = np.zeros(n_samples, dtype=bool)
    known_labels = set()
    class_counts = {}
    classified_indices = []
    hit_log = []
    human_labels_by_idx = df["human_labels"].values
    label_centroid_cache: dict[str, dict[str, np.ndarray | int]] = {}

    # Groundtruth Bayesian drop-threshold configuration
    GROUNDTRUTH_MIN_USEFUL_HIT_RATE = 0.05
    GROUNDTRUTH_DROP_CONFIDENCE = 0.95

    # Per-class tracking for the drop rule
    attempts_per_class = {}
    hits_per_class = {}
    dropped_classes = set()
    dropped_at_step = {}

    def update_label_centroid_cache(human_labels, idx: int) -> None:
        emb = all_embeddings[idx]
        for label in human_labels:
            state = label_centroid_cache.get(label)
            if state is None:
                label_centroid_cache[label] = {
                    "mean": normalize(emb),
                    "count": 1,
                }
                continue
            prev_mean = state["mean"]
            prev_count = int(state["count"])
            new_count = prev_count + 1
            state["mean"] = normalize(update_mean(prev_mean, emb, new_count))
            state["count"] = new_count

    if (
        initial_classified_indices is not None
        and initial_known_labels is not None
        and initial_class_counts is not None
    ):
        # Start from provided state (e.g. diverse phase output)
        classified_indices = list(initial_classified_indices)
        known_labels = set(initial_known_labels)
        class_counts = dict(initial_class_counts)
        for idx in classified_indices:
            is_classified[idx] = True
            human_labels = human_labels_by_idx[idx]
            update_label_centroid_cache(human_labels, idx)
        num_groundtruth_steps = steps
    else:
        # Warm start: pick one random recording and learn its labels
        first_pick_index = df.sample(1, random_state=seed).index[0]
        first_pick = df.index.get_loc(first_pick_index)
        is_classified[first_pick] = True
        classified_indices.append(first_pick)

        human_labels = human_labels_by_idx[first_pick]
        count_labels(human_labels, known_labels, class_counts)
        update_label_centroid_cache(human_labels, first_pick)

        num_groundtruth_steps = steps - 1  # first was warm start

    for step in range(num_groundtruth_steps):
        next_idx, target_class = get_next_groundtruth(
            all_embeddings,
            all_embeddings_norm,
            is_classified,
            human_labels_by_idx,
            class_counts,
            classified_indices,
            label_centroid_cache,
            window_size,
            dominant_window_size=dominant_window_size,
            dropped_classes=dropped_classes,
        )

        if next_idx is None:
            raise ValueError(f"Groundtruth returned None at step {step}")

        is_classified[next_idx] = True
        classified_indices.append(next_idx)

        human_labels = human_labels_by_idx[next_idx]
        count_labels(human_labels, known_labels, class_counts)
        update_label_centroid_cache(human_labels, next_idx)

        hit = target_class in human_labels
        # human_labels can be a list-like or numpy array; avoid ambiguous truth checks
        if human_labels is None:
            first_label = None
        else:
            first_label = human_labels[0] if len(human_labels) > 0 else None
        hit_log.append((target_class, first_label, hit))

        # Update per-class attempts / hits
        attempts_per_class[target_class] = attempts_per_class.get(target_class, 0) + 1
        if hit:
            hits_per_class[target_class] = hits_per_class.get(target_class, 0) + 1

        attempts = attempts_per_class[target_class]
        hits = hits_per_class.get(target_class, 0)

        # Decide whether to drop this label using Bayesian rule
        if should_drop_label_bayesian(
            hits=hits,
            attempts=attempts,
            min_useful_hit_rate=GROUNDTRUTH_MIN_USEFUL_HIT_RATE,
            confidence=GROUNDTRUTH_DROP_CONFIDENCE,
        ):
            if target_class not in dropped_classes:
                dropped_classes.add(target_class)
                dropped_at_step[target_class] = step + 1

    return classified_indices, known_labels, class_counts, hit_log, dropped_at_step


# ---------------------------------------------------------------------------
# Legacy groundtruth utilities (kept for reference, not used by runners yet)
# ---------------------------------------------------------------------------

# def backfill_counts(df, classified_indices):
#     """Scan all classified recordings and build accurate label counts."""
#     known_labels = set()
#     class_counts = {}
#     for idx in classified_indices:
#         for label in df.iloc[idx]["human_labels"]:
#             if label not in known_labels:
#                 known_labels.add(label)
#                 class_counts[label] = 0
#             class_counts[label] += 1
#     return known_labels, class_counts
#
#
def analyze_groundtruth_hits(hit_log, dropped_at_step=None):
    """Per-class hit/miss breakdown.

    Each hit_log entry is expected to be:
        (target_class, actual_first_label, hit_bool)

    Returns a list of dict rows with:
        target_class, hits, misses, total, hit_rate, cost, miss_detail
    where cost is defined as total / hits (attempts per successful hit).
    """
    from collections import Counter

    if dropped_at_step is None:
        dropped_at_step = {}

    targets = sorted(set(t for t, _, _ in hit_log))
    rows = []

    total_hits = 0
    total_all = 0

    for cls in targets:
        entries = [(t, a, h) for t, a, h in hit_log if t == cls]
        hits = sum(1 for _, _, h in entries if h)
        misses = len(entries) - hits
        total = len(entries)
        hit_rate = hits / total if total > 0 else 0.0
        cost = (total / hits) if hits > 0 else None

        total_hits += hits
        total_all += total

        miss_classes = Counter(a for _, a, h in entries if not h)
        miss_detail = "; ".join(f"{c}: {n}" for c, n in miss_classes.most_common())
        dropped_step = dropped_at_step.get(cls, None)

        rows.append(
            {
                "target_class": cls,
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": round(hit_rate, 4),
                "cost": round(cost, 2) if cost is not None else None,
                "dropped_at": dropped_step if dropped_step is not None else "-",
                "miss_detail": miss_detail if miss_detail else "-",
            }
        )

    overall_rate = total_hits / total_all if total_all > 0 else 0.0
    overall_cost = (total_all / total_hits) if total_hits > 0 else None
    rows.append(
        {
            "target_class": "OVERALL",
            "hits": total_hits,
            "misses": total_all - total_hits,
            "total": total_all,
            "hit_rate": round(overall_rate, 4),
            "cost": round(overall_cost, 2) if overall_cost is not None else None,
            "dropped_at": "-",
            "miss_detail": "-",
        }
    )

    return rows
