from datasets import load_dataset
from pathlib import Path
import soundfile as sf

dataset = load_dataset(
    "agkphysics/AudioSet",
    name="balanced",
    split="test",   # this IS the eval split
)

output_dir = Path("/home/lucaa/urban-noise-classification/audioset/eval_set_flac")
output_dir.mkdir(parents=True, exist_ok=True)

for i, sample in enumerate(dataset):
    audio = sample["audio"]
    sf.write(
        output_dir / f"{i:05d}_{sample['video_id']}.flac",
        audio["array"],
        audio["sampling_rate"],
        format="FLAC",
    )