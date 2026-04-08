import os
import re

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
HIT_DIR = os.path.join(BASE_DIR, "hit_results")

# Configure which run to analyze
STEPS = 8000


def load_seed_hits(steps: int = STEPS) -> dict[int, pd.DataFrame]:
    """Load hit CSVs for the mid-groundtruth config, keyed by seed.

    Discovers all matching files in HIT_DIR rather than assuming
    a fixed seed range, so it works for any number of seeds.
    """
    by_seed: dict[int, pd.DataFrame] = {}

    pattern = rf"^mid_groundtruth_n15_{steps}_hits_seed(\d+)\.csv$"

    for fname in sorted(os.listdir(HIT_DIR)):
        m = re.match(pattern, fname)
        if not m:
            continue
        seed = int(m.group(1))
        path = os.path.join(HIT_DIR, fname)
        by_seed[seed] = pd.read_csv(path)

    if not by_seed:
        raise FileNotFoundError(
            f"No mid_groundtruth_n15_{steps} hit CSVs found in {HIT_DIR}"
        )

    return by_seed


def main() -> None:
    print(f"Loading hit CSVs from: {HIT_DIR}")
    print(f"Steps: {STEPS}\n")
    by_seed = load_seed_hits()
    print(f"Found {len(by_seed)} seeds: {sorted(by_seed.keys())}\n")

    hit_rates = []
    for seed in sorted(by_seed.keys()):
        df = by_seed[seed]

        overall = df[df["target_class"] == "OVERALL"].iloc[0]
        hit_rate_pct = overall["hit_rate"] * 100
        hit_rates.append(round(hit_rate_pct, 2))

        # Count dropped classes
        per_class = df[df["target_class"] != "OVERALL"].copy()
        dropped_count = (per_class["dropped_at"] != "-").sum()

        print(f"=== Seed {seed} ===")
        print(
            f"OVERALL: hits={int(overall['hits'])}, "
            f"misses={int(overall['misses'])}, "
            f"total={int(overall['total'])}, "
            f"hit%={hit_rate_pct:.2f}, "
            f"dropped={dropped_count}/{len(per_class)}"
        )

        # Sort by hit_rate descending (best first)
        per_class = per_class.sort_values("hit_rate", ascending=False).reset_index(drop=True)

        print(
            "\nclass                                    hits   misses    hit%     cost  dropped_at\n"
            "---------------------------------------------------------------------------------"
        )
        for _, row in per_class.iterrows():
            cls = row["target_class"]
            cls_disp = (cls[:36] + "...") if len(cls) > 39 else cls
            cost = row["cost"] if "cost" in row and pd.notna(row["cost"]) else None
            cost_str = f"{cost:6.2f}" if cost is not None else "   -  "
            dropped_at = row.get("dropped_at", "-")
            dropped_str = f"{dropped_at:>10}" if dropped_at != "-" else "         -"
            print(
                f"{cls_disp:40s}  "
                f"{int(row['hits']):4d}  "
                f"{int(row['misses']):7d}  "
                f"{row['hit_rate']*100:6.2f}  "
                f"{cost_str}  "
                f"{dropped_str}"
            )

        print("\n")

    print(f"Hit % by seed: {hit_rates}")
    print(f"Mean hit %: {np.mean(hit_rates):.2f}")


if __name__ == "__main__":
    main()

