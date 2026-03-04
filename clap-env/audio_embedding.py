import glob
import os

import laion_clap
import librosa
import numpy as np


# Compatibility workaround: np.random.integers was removed in newer NumPy versions
# Because of new environment setup added code:
if not hasattr(np.random, "integers"):

    def integers(low, high=None, size=None, dtype=np.int64):
        rng = np.random.default_rng()
        return rng.integers(low, high, size=size, dtype=dtype)

    np.random.integers = integers

INPUT_DIR = (
    "/home/arriburi/projects/urban-noise-classification/audioset/eval_set_flac"
)
OUTPUT_NPZ = "/home/arriburi/projects/urban-noise-classification/clap-env/clap_audio_embeddings.npz"


def find_audio_files(root: str) -> list[str]:
    patterns = [
        os.path.join(root, "**", "*.wav"),
        os.path.join(root, "**", "*.flac"),
    ]
    files: set[str] = set()
    for pattern in patterns:
        files.update(glob.glob(pattern, recursive=True))
    return sorted(files)


def extract_video_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem  ## just return after _


model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
model.load_ckpt(
    "/home/arriburi/projects/urban-noise-classification/clap-env/music_speech_audioset_epoch_15_esc_89.98.pt"
)

audio_files = find_audio_files(INPUT_DIR)

if not audio_files:
    print(f"No audio files found under {INPUT_DIR}")
    exit(1)

print(f"Found {len(audio_files)} audio files to process")

batch_size = 100
all_embeddings: list[np.ndarray] = []

total_items = len(audio_files)
for i in range(0, len(audio_files), batch_size):
    batch_files = audio_files[i : i + batch_size]
    items_processed = min(i + batch_size, total_items)

    if items_processed % 100 == 0:
        print(f"Processing batches {items_processed}/{total_items}")

    try:
        audio_batch = []
        for file_path in batch_files:
            audio_data, _ = librosa.load(file_path, sr=48000, mono=True)
            audio_batch.append(audio_data)

        max_len = max(a.shape[0] for a in audio_batch)
        audio_batch = np.stack(
            [
                np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a[:max_len]
                for a in audio_batch
            ]
        ).astype(np.float32)

        batch_embeddings = model.get_audio_embedding_from_data(
            x=audio_batch, use_tensor=False
        )
        all_embeddings.append(batch_embeddings)
    except Exception as e:
        print(f"Error processing batch: {e}")
        import traceback

        traceback.print_exc()
        exit(1)

audio_embed = np.vstack(all_embeddings)
paths_array = np.asarray(audio_files)
video_ids = np.asarray([extract_video_id(path) for path in audio_files])
print(f"Final embeddings shape: {audio_embed.shape}")

np.savez(
    OUTPUT_NPZ,
    embeddings=audio_embed,
    paths=paths_array,
    video_ids=video_ids,
)
print(f"Saved embeddings + paths + video_ids to: {OUTPUT_NPZ}")
