from datasets import load_dataset, Audio
from pathlib import Path
import soundfile as sf
import io

dataset = load_dataset(
    "agkphysics/AudioSet",
    name="balanced",
    split="test",  # this IS the eval split
)

# Avoid TorchCodec: get raw bytes/paths and decode with soundfile.
dataset = dataset.cast_column("audio", Audio(decode=False))

output_dir = Path(
    "/home/arriburi/projects/urban-noise-classification/audioset/eval_set_flac"
)
output_dir.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset):
    audio = sample["audio"]

    # audio is {"bytes": ..., "path": ...}; prefer bytes if present.
    if audio.get("bytes") is not None:
        data, sr = sf.read(io.BytesIO(audio["bytes"]))
    else:
        data, sr = sf.read(audio["path"])

    sf.write(
        output_dir / f"{i:05d}_{sample['video_id']}.flac",
        data,
        sr,
        format="FLAC",
    )
