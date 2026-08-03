import argparse
import glob
import os
from pathlib import Path

import laion_clap
import librosa
import numpy as np


# Compatibility workaround: np.random.integers was removed in newer NumPy versions
if not hasattr(np.random, "integers"):

    def integers(low, high=None, size=None, dtype=np.int64):
        rng = np.random.default_rng()
        return rng.integers(low, high, size=size, dtype=dtype)

    np.random.integers = integers


DEFAULT_INPUT_DIR = "C:/Users/Luca/Desktop/urban-noise-classification/audioset/eval_set_flac"
DEFAULT_OUTPUT_NPZ = (
    "C:/Users/Luca/Desktop/urban-noise-classification/clap-env/clap_audio_embeddings.npz"
)
DEFAULT_CKPT = (
    "C:/Users/Luca/Desktop/urban-noise-classification/"
    "clap-env/music_speech_audioset_epoch_15_esc_89.98.pt"
)
DEFAULT_AMODEL = "auto"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CLAP embeddings for raw AudioSet audio.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_NPZ)
    parser.add_argument("--clap-ckpt", default=DEFAULT_CKPT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amodel", default=DEFAULT_AMODEL)
    parser.add_argument(
        "--enable-fusion",
        action="store_true",
        help="Enable CLAP feature fusion. If omitted, fusion is auto-enabled for checkpoints with 'fusion' in the filename.",
    )
    return parser.parse_args()


def _resolve_enable_fusion(clap_ckpt: str, enable_fusion: bool) -> bool:
    if enable_fusion:
        return True
    return "fusion" in Path(clap_ckpt).stem.lower()


def _resolve_amodel(clap_ckpt: str, amodel: str) -> str:
    if amodel != DEFAULT_AMODEL:
        return amodel

    ckpt_name = Path(clap_ckpt).stem.lower()
    if ckpt_name.startswith("630k"):
        return "HTSAT-tiny"
    if ckpt_name.startswith("music_"):
        return "HTSAT-base"
    return "HTSAT-base"


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
    return parts[1] if len(parts) == 2 else stem


def main() -> None:
    args = parse_args()
    enable_fusion = _resolve_enable_fusion(args.clap_ckpt, args.enable_fusion)
    amodel = _resolve_amodel(args.clap_ckpt, args.amodel)

    print("Loading CLAP model...")
    print(f"  checkpoint: {args.clap_ckpt}")
    print(f"  amodel: {amodel}")
    print(f"  enable_fusion: {enable_fusion}")
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion, amodel=amodel)
    model.load_ckpt(args.clap_ckpt)

    audio_files = find_audio_files(args.input_dir)
    if not audio_files:
        print(f"No audio files found under {args.input_dir}")
        raise SystemExit(1)

    print(f"Found {len(audio_files)} audio files to process")

    all_embeddings: list[np.ndarray] = []
    total_items = len(audio_files)
    for i in range(0, len(audio_files), args.batch_size):
        batch_files = audio_files[i : i + args.batch_size]
        items_processed = min(i + args.batch_size, total_items)

        if items_processed % 100 == 0 or items_processed == total_items:
            print(f"Processing batches {items_processed}/{total_items}")

        try:
            audio_batch = []
            for file_path in batch_files:
                audio_data, _ = librosa.load(file_path, sr=48000, mono=True)
                audio_batch.append(audio_data)

            max_len = max(a.shape[0] for a in audio_batch)
            audio_batch_np = np.stack(
                [
                    np.pad(a, (0, max_len - len(a))) if len(a) < max_len else a[:max_len]
                    for a in audio_batch
                ]
            ).astype(np.float32)

            batch_embeddings = model.get_audio_embedding_from_data(
                x=audio_batch_np, use_tensor=False
            )
            all_embeddings.append(batch_embeddings)
        except Exception as exc:
            print(f"Error processing batch: {exc}")
            import traceback

            traceback.print_exc()
            raise SystemExit(1)

    audio_embed = np.vstack(all_embeddings)
    paths_array = np.asarray(audio_files)
    video_ids = np.asarray([extract_video_id(path) for path in audio_files])
    print(f"Final embeddings shape: {audio_embed.shape}")

    np.savez(
        args.output,
        embeddings=audio_embed,
        paths=paths_array,
        video_ids=video_ids,
    )
    print(f"Saved embeddings + paths + video_ids to: {args.output}")


if __name__ == "__main__":
    main()
