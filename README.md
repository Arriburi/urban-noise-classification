# Urban Noise Classification Pipeline

## Pipeline Order

1. **Download eval audio files**
   ```bash
   python audioset/download_eval.py
   ```
   → Outputs FLAC files to `audioset/eval_set_flac/`

2. **Generate CLAP audio embeddings (required)**
   ```bash
   python clap-env/audio_embedding.py
   ```
   → Outputs `clap-env/clap_audio_embeddings.npz`

3. **(Optional) Generate CLAP text embeddings**
   ```bash
   python clap-env/text_embedding.py
   ```
   → Outputs `clap-env/clap_text_embeddings.npz` (only needed for experiments using text–audio similarity)

4. **Create base parquet dataset (combine embeddings, paths, video IDs, labels)**
   - Requires: `audioset/eval_segments.csv` from the official AudioSet eval metadata.
   ```bash
   python audioset/parquet_creator.py
   ```
   → Outputs `audioset/audioset_eval.parquet`

5. **Transform labels to SALT mid-layer (for simulations)**
   ```bash
   python clap-env/simulation/transform_audioset.py
   ```
   → Reads `audioset/audioset_eval.parquet`, writes `clap-env/simulation/audioset_eval_mid.parquet`

6. **Run active-learning simulations (all strategies / seeds)**
   ```bash
   python clap-env/simulation/run_labeling.py
   ```
   → Outputs per-seed simulation results under `clap-env/simulation/outputs/`

7. **Analyze distributions (entropy, coverage, imbalance, shift)**
   ```bash
   python clap-env/simulation/check_distribution.py
   ```

8. **(Optional) Analyze groundtruth hits and SALT-based matches**
   ```bash
   python clap-env/simulation/analyze_hits.py
