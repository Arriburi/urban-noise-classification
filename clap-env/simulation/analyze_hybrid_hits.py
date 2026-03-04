import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(__file__)
HIT_DIR = os.path.join(BASE_DIR, "hit_results")


def main() -> None:
    # Parse total steps and diverse steps from filename:
    #   mid_hybrid_<TOTAL>_d<DIVERSE>_n15_hits_seed<SEED>.csv
    pattern = re.compile(r"^mid_hybrid_(\d+)_d(\d+)_n15_hits_seed(\d+)\.csv$")

    # Keyed by (diverse_steps, groundtruth_steps)
    by_config: dict[tuple[int, int], list[tuple[int, str, int]]] = defaultdict(list)

    for fname in sorted(os.listdir(HIT_DIR)):
        m = pattern.match(fname)
        if not m:
            continue
        total_steps = int(m.group(1))
        d_steps = int(m.group(2))
        seed = int(m.group(3))
        gt_steps = total_steps - d_steps
        path = os.path.join(HIT_DIR, fname)
        by_config[(d_steps, gt_steps)].append((seed, path, total_steps))

    if not by_config:
        print(f"No hybrid hit CSVs matching {pattern.pattern} found in {HIT_DIR}")
        return

    # Sort by diverse steps, then groundtruth steps (both ascending)
    for (d_steps, gt_steps) in sorted(by_config.keys(), key=lambda x: (x[0], x[1])):
        entries = sorted(by_config[(d_steps, gt_steps)], key=lambda x: x[0])
        seeds = []
        hit_pcts = []
        # All entries in this group share the same total steps
        total_steps = entries[0][2]

        for seed, path, _ in entries:
            df = pd.read_csv(path)
            overall = df[df["target_class"] == "OVERALL"].iloc[0]
            hit_pct = float(overall["hit_rate"]) * 100.0
            seeds.append(seed)
            hit_pcts.append(hit_pct)

        arr = np.array(hit_pcts)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        print(
            f"=== Hybrid d{d_steps} + gt{gt_steps} (total {total_steps} steps) ==="
        )
        print(f"Seeds: {seeds}")
        print("Hit % by seed:", [round(x, 2) for x in hit_pcts])
        print(f"Mean hit %: {mean:.2f} (std {std:.2f})\n")


if __name__ == "__main__":
    main()

