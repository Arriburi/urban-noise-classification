#!/usr/bin/env python3
import os
import sys

import pandas as pd

# Use the local py-salt repo under clap-env/salt/py-salt
SALT_REPO = os.path.join(os.path.dirname(__file__), "..", "salt", "py-salt")
if SALT_REPO not in sys.path:
    sys.path.append(SALT_REPO)

from py_salt import event_mapping

e = event_mapping.EventExplorer()

# Use the audioset_eval.parquet created by audioset/parquet_creator.py
ORIGINAL_PARQUET = (
    "/home/arriburi/projects/urban-noise-classification/audioset/audioset_eval.parquet"
)
SIMULATION_DIR = os.path.dirname(__file__)


def to_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        return list(value)
    except TypeError:
        return [value]


# Simple in-process cache so we only resolve each distinct label once.
_PATH_CACHE = {}

# SALT top-layer labels for this taxonomy (depth 1 roots).
TOP_LEVEL_LABELS = {
    "human_sounds",
    "animal",
    "music",
    "natural_sounds",
    "source-ambiguous_sounds",
    "channel_environment_and_background",
    "sound_of_things",
    "water",
    "human_activities",
    "other",
}


def get_all_paths(label):
    """Get ALL SALT ontology paths for a label.

    Returns list of paths (each a list of labels) or None when the
    label cannot be mapped into the SALT taxonomy.
    """
    key = str(label).strip()
    if key in _PATH_CACHE:
        return _PATH_CACHE[key]

    try:
        std = e.get_std_label_from_dataset_label(key)
        paths = e.get_paths_to_label(std)
    except Exception:
        paths = None
    if not paths:
        paths = None

    _PATH_CACHE[key] = paths
    return paths


def extract_mid_layer_from_paths(paths):
    """Extract mid-layer labels from all paths.

    Mid-layer is depth 2 (path[1] if path length > 1, else path[0]).
    Returns set of unique mid-layer labels.
    """
    if paths is None:
        return set()

    mid_layers = set()
    for path in paths:
        if not path:
            continue
        if len(path) == 1:
            # Top-level only label
            mid_layers.add(path[0])
        else:
            # Mid-layer is at index 1 (depth 2)
            mid_layers.add(path[1])

    return mid_layers


def coarsen_to_mid_layer(label):
    """Map a label to its mid-layer parent(s), handling multiple paths.

    Returns set of mid-layer labels (or None if label can't be mapped).
    For labels with multiple paths, returns ALL mid-layer parents.
    """
    paths = get_all_paths(label)
    if paths is None:
        return None
    return extract_mid_layer_from_paths(paths)


def prune_redundant_top_layers(labels):
    """Given a list of (already mid-layer) SALT labels, drop any top-layer
    label that has a more specific descendant also present in the list.

    Example:
        ['domestic_animal', 'onomatopoeia', 'animal']
        -> ['domestic_animal', 'onomatopoeia']   # drop 'animal'

    Assumes labels have already been coarsened to mid-layer / top-layer.
    """
    if not labels:
        return labels

    labels = list(labels)
    top_labels = [label for label in labels if label in TOP_LEVEL_LABELS]
    if not top_labels:
        return labels

    # Pre-fetch paths for all labels once (using cached get_all_paths).
    paths_by_label = {label: get_all_paths(label) for label in labels}

    to_drop = set()
    for top in top_labels:
        top_paths = paths_by_label.get(top)
        # If we can't resolve the top label in SALT, be conservative and keep it.
        if not top_paths:
            continue

        # A proper descendant must have a path that starts with the top label
        # and is longer than length 1.
        is_redundant = False
        for other, other_paths in paths_by_label.items():
            if other == top or not other_paths:
                continue

            for op in other_paths:
                if not op or len(op) <= 1:
                    continue
                # SALT paths are root -> leaf, so top is an ancestor if it is
                # the first element and there is at least one more level.
                if op[0] == top:
                    is_redundant = True
                    break
            if is_redundant:
                break

        if is_redundant:
            to_drop.add(top)

    if not to_drop:
        return labels

    return [label for label in labels if label not in to_drop]


def create_mid_parents_file():
    """Create parquet file with labels mapped to mid-layer (depth 2).

    For each recording:
    1. Map all labels to their mid-layer parents (handling multiple paths)
    2. Keep top-level labels that have no deeper node
    3. Deduplicate labels
    """
    df = pd.read_parquet(ORIGINAL_PARQUET)

    unmapped_labels = set()

    def transform_labels(label_list):
        nonlocal unmapped_labels

        labels = to_list(label_list)

        # Collect all mid-layer labels from all paths
        mid_layer_set = set()
        for label in labels:
            paths = get_all_paths(label)
            if not paths:
                unmapped_labels.add(str(label))
                continue

            for path in paths:
                if not path:
                    continue
                if len(path) == 1:
                    mid_layer_set.add(path[0])
                else:
                    mid_layer_set.add(path[1])

        pruned = prune_redundant_top_layers(mid_layer_set)
        return list(pruned)

    df["human_labels"] = df["human_labels"].apply(transform_labels)

    if unmapped_labels:
        raise RuntimeError(
            "Failed to map the following labels into SALT mid-layer: "
            + ", ".join(sorted(unmapped_labels))
        )

    output_path = os.path.join(SIMULATION_DIR, "audioset_eval_mid.parquet")
    df.to_parquet(output_path, index=False)

    # Print statistics
    all_labels = set()
    for labels in df["human_labels"]:
        all_labels.update(labels)
    print(f"Transformed to {len(all_labels)} unique mid-layer classes")
    print("Top 10 classes by count:")
    from collections import Counter

    counts = Counter()
    for labels in df["human_labels"]:
        for label in labels:
            counts[label] += 1
    for label, count in counts.most_common(10):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    # Always run the mid-layer transformation when executed as a script.
    create_mid_parents_file()
