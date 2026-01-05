# Urban Noise Classification Pipeline

## Pipeline Order

1. **Download eval audio files**
   ```bash
   uv run audioset/download_eval.py
   ```
   → Outputs FLAC files to `eval_set_flac/`

2. **Generate CLAP audio and text embeddings**
   ```bash
   uv run clap-env/audio_embedding.py
   uv run clap-env/text_embedding.py
   ```
   → Outputs `clap_audio_embeddings.npz`

3. **Create parquet dataset -> combine embeddings and recording paths + video ids**
   ```bash
   uv run audioset/parquet_creator.py
   ```
   → Outputs `audioset_eval.parquet`

4. **Create simulation results in various mode strategies**
   ```bash
   uv run simulation/run_labeling.py
   ```
   → Outputs `simulation_results_{mode}_{steps}.parquet`

5. **Compare clap and audioset labels for matches**
   ```bash
   uv run simulation/postprocess_salt.py
   ```
   → Outputs `salt_matches.parquet`
