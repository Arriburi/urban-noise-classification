import os
import re

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
HIT_DIR = os.path.join(BASE_DIR, "hit_results")


def load_seed_hits() -> dict[int, pd.DataFrame]:
    """Load hit CSVs for the mid-groundtruth config, keyed by seed."""
    pattern = r"^mid_groundtruth_n15_2000_hits_seed(\d+)\.csv$"
    by_seed: dict[int, pd.DataFrame] = {}

    for fname in sorted(os.listdir(HIT_DIR)):
        m = re.match(pattern, fname)
        if not m:
            continue
        seed = int(m.group(1))
        path = os.path.join(HIT_DIR, fname)
        by_seed[seed] = pd.read_csv(path)

    if not by_seed:
        raise FileNotFoundError(
            f"No mid_groundtruth_n15_2000 hit CSVs found in {HIT_DIR}"
        )

    return by_seed


def main() -> None:
    print(f"Loading hit CSVs from: {HIT_DIR}")
    by_seed = load_seed_hits()
    print(f"Found {len(by_seed)} seeds: {sorted(by_seed.keys())}\n")

    hit_rates = []
    for seed in sorted(by_seed.keys()):
        df = by_seed[seed]

        overall = df[df["target_class"] == "OVERALL"].iloc[0]
        hit_rate_pct = overall["hit_rate"] * 100
        hit_rates.append(round(hit_rate_pct, 2))

        print(f"=== Seed {seed} ===")
        print(
            f"OVERALL: hits={int(overall['hits'])}, "
            f"misses={int(overall['misses'])}, "
            f"total={int(overall['total'])}, "
            f"hit%={hit_rate_pct:.2f}"
        )

        per_class = df[df["target_class"] != "OVERALL"].copy()
        per_class = per_class.sort_values("hit_rate").reset_index(drop=True)

        print(
            "\nclass                                    hits   misses    hit%     cost\n"
            "---------------------------------------------------------------------"
        )
        for _, row in per_class.iterrows():
            cls = row["target_class"]
            cls_disp = (cls[:36] + "...") if len(cls) > 39 else cls
            cost = row["cost"] if "cost" in row and not np.isnan(row["cost"]) else None
            cost_str = f"{cost:6.2f}" if cost is not None else "   -  "
            print(
                f"{cls_disp:40s}  "
                f"{int(row['hits']):4d}  "
                f"{int(row['misses']):7d}  "
                f"{row['hit_rate']*100:6.2f}  "
                f"{cost_str}"
            )

        print("\n")

    print(f"Hit % by seed: {hit_rates}")


if __name__ == "__main__":
    main()

