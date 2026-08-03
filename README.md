# Urban Noise Classification - Active Learning Simulations

This repository contains the active learning simulation framework for evaluating different annotation/labeling strategies (e.g., Random, Diversity-based, K-Means Clustering, HDBSCAN, and balancing-guided methods) for urban noise classification using precomputed CLAP audio embeddings.

---

## 1. Input Data Requirements

The preprocessed dataset Parquet files used in our simulations (such as `esc50.parquet`) are excluded from this repository due to their size. To run the simulation pipeline, you must provide your own locally preprocessed Parquet file(s).

Each input Parquet file **must** contain the following standardized schema:
* `video_id` *(string)*: Unique identifier for each recording (e.g., audio filename).
* `human_labels` *(list of strings)*: The ground-truth labels assigned to the recording (e.g., `["Dog", "Bark"]` or `None`).
* `embedding` *(NumPy array or list of floats)*: The precomputed CLAP feature embedding (typically a 1D vector generated using a `laion_clap` library HTSAT model).

---

## 2. Configuring & Reproducing Simulations

All simulation settings, dataset selections, and hyperparameters are configured directly inside [run_labeling.py](file:///c:/Users/Luca/Desktop/urban-noise-classification/clap-env/simulation/run_labeling.py). Use the following parameters to configure runs or reproduce our thesis experiments:

* **Dataset Selection**: Define the path to your local `.parquet` file in the `PARQUET_OPTIONS` dictionary, then add its key to the `ACTIVE_PARQUETS` list (e.g., `ACTIVE_PARQUETS = ["esc50"]`).
* **Active Strategies**: Modify the `MODES` list to include strategies you want to run (options: `"random"`, `"diverse"`, `"kmeans"`, `"balanced_partition"`, `"hdbscan"`, `"groundtruth"`, `"groundtruth_cold"`, `"hybrid"`).
* **Seeds & Parallelism**: Adjust `seeds = range(1, 31)` to run **30 distinct seeds**. Calculating the mean and standard deviation across these 30 trials is required to account for stochastic components in active strategies (e.g., K-Means initialization or random sampling). Set `num_workers = 8` to specify CPU cores for parallel execution.
* **Start Configurations**:
  * *Seeded Start*: Pre-seeds the simulation with a small set of labeled examples per class (e.g., `10` examples per class for AudioSet mid-layer, `3` examples per class for ESC-50). These pre-seeded samples do not count towards the steps budget.
  * *Cold Start*: Starts from a single randomly selected sample, without any prior seeded annotations (e.g., run the `"groundtruth_cold"` strategy).
* **Number of Steps**: Adjust the steps budget per dataset in the `DATASET_TOTAL_STEPS` dictionary.
* **DLR (Dominant Label Repulsion) Parameter $k$**: Define your ablation configurations inside `GROUNDTRUTH_EXPERIMENTS` by specifying `top_k` values.
  * *AudioSet (mid-layer)*: Set `top_k = 2` (calibrated as the optimal threshold for both seeded and cold-start experiments to maximize entropy).
  * *ESC-50 & UrbanSound*: Set `top_k = 1` (a neutral, robust choice for single-label datasets to prevent excessive distribution oscillations).
* **Bayesian Drop**: Toggle `enable_bayesian_drop` (inside `GROUNDTRUTH_EXPERIMENTS`) to enable/disable Bayesian dynamic class-skipping rules.

---

## 3. Running the Simulations

Run the simulation manager script using:

```bash
python clap-env/simulation/run_labeling.py
```

### What happens when run:
1. The script loads the selected dataset parquet and pre-normalizes the CLAP embeddings for cosine similarity.
2. It parallelizes the execution across your workers.
3. For each seed and strategy, the active selection loop runs until the dataset's step limit is reached.
4. Each run's history is recorded and written to disk.

---

## 4. Simulation Outputs

The simulation produces two types of output logs (stored by default inside `clap-env/simulation/thesis_seeded/`):

### A. Chronological Sampling Logs (Parquet Format)
Saved for **all** strategies inside `simulation_outputs/`. Each file records the exact sequence of selected samples step-by-step. These logs can then be used for various downstream testing (such as measuring entropy, hit/miss ratio, label coverage over time, or top-k performance).

* **Filename format**: e.g., `esc50_random_235_seed1.parquet` or `esc50_kmeans_random_km22_235_seed1.parquet`.
* **Structure**:
  * `video_id` *(string)*: Unique identifier of the selected audio sample.
  * `human_labels` *(list of strings)*: Ground-truth labels associated with that sample.

### B. Balancing Hit & Miss Logs (CSV Format)
Saved only for **balancing-guided methods** (e.g., `groundtruth`, `groundtruth_cold`, `hybrid`) inside `hit_results/` when `ENABLE_HIT_ANALYSIS = True`. These logs record the class-targeting statistics and drop details of the simulation.

* **Filename format**: e.g., `esc50_groundtruth_k2_235_hits_seed1.csv`.
* **Structure**:
  * `target_class` *(string)*: The class that the strategy attempted to target.
  * `hits` *(int)*: Successful attempts to sample this class.
  * `misses` *(int)*: Unsuccessful attempts (where a different class was sampled).
  * `total` *(int)*: Total attempts made targeting this class.
  * `hit_rate` *(float)*: Ratio of hits to total.
  * `cost` *(float)*: Average attempts per successful hit (`total / hits`).
  * `dropped_at` *(int or '-')*: The simulation step at which this class was dropped (due to Bayesian rules).
  * `miss_detail` *(string)*: Semicolon-separated breakdown of which wrong classes were actually sampled instead (e.g., `Dog: 3; Bark: 2`).
