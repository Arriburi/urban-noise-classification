import json
from pathlib import Path

import numpy as np
import pandas as pd


## to use after audio_embedding.py has been run

CSV_PATH = Path(
    "/home/arriburi/projects/urban-noise-classification/audioset/eval_segments.csv"
)
FLAC_DIR = Path(
    "/home/arriburi/projects/urban-noise-classification/audioset/eval_set_flac"
)
ONTOLOGY_PATH = Path(
    "/home/arriburi/projects/urban-noise-classification/clap-env/ontology.json"
)
AUDIO_EMBEDDINGS_PATH = Path(
    "/home/arriburi/projects/urban-noise-classification/clap-env/clap_audio_embeddings.npz"
)
OUTPUT_PATH = Path(
    "/home/arriburi/projects/urban-noise-classification/audioset/audioset_eval.parquet"
)


def load_ontology_mapping() -> dict[str, str]:
    with open(ONTOLOGY_PATH, "r") as f:
        ontology = json.load(f)
    mapping = {}
    for entry in ontology:
        entry_id = entry.get("id")
        name = entry.get("name", "").strip()
        mapping[entry_id] = name
    return mapping


def load_audio_embeddings() -> pd.DataFrame:
    npz_data = np.load(AUDIO_EMBEDDINGS_PATH, allow_pickle=True)
    return pd.DataFrame(
        {
            "file_path": npz_data["paths"].tolist(),
            "embedding": npz_data["embeddings"].tolist(),
            "video_id": npz_data["video_ids"].tolist(),
        }
    )


def map_labels_to_names(label_ids: list[str], id_to_name: dict[str, str]) -> list[str]:
    results: list[str] = []
    for label in label_ids:
        name = id_to_name.get(label, label)
        results.append(name)
    return results


def main() -> None:
    id_to_name = load_ontology_mapping()
    audio_embeddings_df = load_audio_embeddings().drop(columns=["file_path"])

    df = pd.read_csv(
        CSV_PATH,
        comment="#",
        header=None,
        names=["video_id", "start_seconds", "end_seconds", "positive_labels"],
        engine="python",
        skipinitialspace=True,
    )

    available = {p.stem.split("_", 1)[1]: p for p in FLAC_DIR.glob("*.flac")}
    df = df[df["video_id"].isin(available)].copy()
    df["file_path"] = df["video_id"].map(lambda vid: available[vid].as_posix())
    df["human_labels"] = (
        df["positive_labels"]
        .str.strip()
        .str.strip('"')
        .str.split(",")
        .apply(lambda labels: [label.strip() for label in labels if label])
        .apply(lambda ids: map_labels_to_names(ids, id_to_name))
    )
    df["clap_labels"] = [None] * len(df)
    df["clap_score"] = [None] * len(df)
    df["is_classified"] = False

    df = df.merge(audio_embeddings_df, on="video_id", how="left")
    df = df[
        [
            "video_id",
            "file_path",
            "embedding",
            "human_labels",
            "clap_labels",
            "clap_score",
            "is_classified",
        ]
    ]
    df.to_parquet(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
