import pandas as pd
import subprocess
import os
import shutil

pd.set_option("display.max_colwidth", None)

temp_dir = "/home/lucaa/urban-noise-classification/audioset/tmp_wav"
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
os.makedirs(temp_dir)

def flac_to_wav(flac_path):
    filename = os.path.basename(flac_path).replace('.flac', '.wav')
    wav_path = os.path.join(temp_dir, filename)
    if not os.path.exists(wav_path):
        subprocess.run(["ffmpeg", "-i", flac_path, "-y", wav_path], 
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    return wav_path

df = pd.read_parquet("/home/lucaa/urban-noise-classification/audioset/audioset_eval.parquet")

display_df = df.sample(5).copy()
display_df["wav_path"] = display_df["file_path"].apply(flac_to_wav)

print(display_df[["human_labels", "wav_path"]].to_string(index=False))