import json
import os
from typing import Iterable, List, Sequence, Tuple

import laion_clap
import numpy as np

BATCH_SIZE = 64

def build_prompt(entry: dict) -> Tuple[str, str]:
    name = entry.get("name", "").strip()
    description = entry.get("description", "").strip()
    if not name:
        raise ValueError(f"Ontology entry is missing a name.")
    if not description:
        raise ValueError(f"Ontology entry is missing a description.")
    return (name, f"{name}: {description}")


def batched(seq: Sequence[str], batch_size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(seq), batch_size):
        yield seq[start : start + batch_size]


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()

ontology_file = os.path.join(os.path.dirname(__file__), "ontology.json")
with open(ontology_file, "r") as f:
    ontology = json.load(f)

# Filter out abastract and blacklist
restrictions_to_exclude = {"abstract"}
filtered_entries = [
    entry
    for entry in ontology
    if not restrictions_to_exclude.intersection(entry.get("restrictions") or [])
]

excluded_abstract = sum(
    1 for entry in ontology if "abstract" in (entry.get("restrictions") or [])
)
excluded_blacklist = sum(
    1 for entry in ontology if "blacklist" in (entry.get("restrictions") or [])
)

print(f"Total ontology entries: {len(ontology)}")
print(f"Entries excluded for 'abstract': {excluded_abstract}")
print(f"Entries excluded for 'blacklist': {excluded_blacklist}")
print(f"Entries kept: {len(filtered_entries)}")

prompt_pairs = [build_prompt(entry) for entry in filtered_entries]
if not prompt_pairs:
    raise RuntimeError("No valid prompts were produced from ontology.json.")

names, text_prompts = zip(*prompt_pairs)

embeddings: List[np.ndarray] = []
for batch in batched(list(text_prompts), BATCH_SIZE):
    try:
        batch_embeddings = model.get_text_embedding(list(batch), use_tensor=False)
    except Exception:
        exit(1)
    embeddings.append(np.asarray(batch_embeddings))

text_embeddings = np.concatenate(embeddings, axis=0)
names_array = np.asarray(names)

embeddings_path = os.path.join(os.path.dirname(__file__), "clap_text_embeddings.npz")
np.savez(
    embeddings_path,
    names=names_array,
    embeddings=text_embeddings,
)
print(f"Saved text embeddings to: {embeddings_path}")

preview = np.load(embeddings_path)
sample_size = min(5, preview["names"].shape[0])
sample_indices = np.random.choice(preview["names"].shape[0], size=sample_size, replace=False)
print(f"Sample names: {preview['names'][sample_indices]}")
