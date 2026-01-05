import os
import sys
from joblib import Parallel, delayed

import pandas as pd

SYS_SALT_ROOT = "/home/lucaa/urban-noise-classification/clap-env"
if SYS_SALT_ROOT not in sys.path:
    sys.path.append(SYS_SALT_ROOT)

from salt import compare_labels  


SALT_MAX_HOPS = 2


def compute_salt_result(clap_label, human_labels):
    try:
        match, results = compare_labels(clap_label, human_labels, max_hops=SALT_MAX_HOPS)

        target = next((r for r in results if r["matched"]), None)
        lca = target.get("lca") if target else None
        return (bool(match), lca)

    except ValueError:
        return (None, None)
    
    except Exception as e:
        print(f"Unexpected error for CLAP: '{clap_label}': {e}")
        return (None, None)


def add_salt_columns(input_path, output_path=None):
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df)} rows from {input_path}", flush=True)

    unique_combos = {}
    for _, row in df.iterrows():
        key = (row["clap_labels"], tuple(row["human_labels"]))
        if key not in unique_combos:
            unique_combos[key] = (row["clap_labels"], row["human_labels"])

    # PARALLEL COMPUTATION
    tasks = []
    for clap_label, human_labels in unique_combos.values():
        task = delayed(compute_salt_result)(clap_label, human_labels)
        tasks.append(task)
    results = Parallel(n_jobs=-1)(tasks)
    
    cache = {}
    unique_keys = list(unique_combos.keys())
    for i in range(len(unique_keys)):
        key = unique_keys[i]       
        result = results[i]        
        cache[key] = result

    salt_match = []
    salt_lca = []

    for _, row in df.iterrows():
        key = (row["clap_labels"], tuple(row["human_labels"]))
        match, lca = cache[key]
        salt_match.append(match)
        salt_lca.append(lca)

    df["salt_match"] = salt_match
    df["salt_lca"] = salt_lca

    print(f"\nUnique combinations: {len(cache)}")

    # Save
    print(f"Saving to: {output_path}", flush=True)
    df.to_parquet(output_path, index=False)
    
    return df


def main():
    INPUT_PATH = "/home/lucaa/urban-noise-classification/clap-env/simulation/simulation_results_similar_5000.parquet"
    OUTPUT_PATH = "/home/lucaa/urban-noise-classification/clap-env/simulation/salt_matches.parquet"

    add_salt_columns(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()


