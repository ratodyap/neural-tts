import os
import json
import librosa 
import numpy as np 
import pandas as pd 
import tqdm
from text_normalize import normalize_text

SR  = 22050
N_MELS = 80 
HOP = 256 
FFT = 1024 

TRAIN_META = "data/train_metadata.csv"
SPK2ID = "data/spk2id.json"
PROCESSED = "data/processed"
OUT_MELS = "data/processed/mels"
OUT_META = "data/processed/mel_metadata.csv"

os.makedirs(OUT_MELS, exist_ok=True)

def wav_to_mel(path):
    y, _ = librosa.load(path, sr =SR)
    mel = librosa.feature.melspectrogram(
        y =y, 
        sr = SR,
        n_fft = FFT,
        hop_length= HOP,
        n_mels = N_MELS,
        fmin = 0,
        fmax = 8000
    )
    mel_db = librosa.power_to_db( mel , ref = np.max)

    return mel_db.T

# read metadata
df = pd.read_csv(TRAIN_META,
                 sep = "|",
                 names = ["file","text","speaker"])
spk2id = json.load(open(SPK2ID))
                
rows_out = []

# Generating Mel - Spectrogram 

for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    spk = row["speaker"]
    wav_name = os.path.basename(row["file"])
    wav_path = f"{PROCESSED}/{spk}/{wav_name}"

    if not os.path.exists(wav_path):
        continue

    mel = wav_to_mel(wav_path)
    mel_file = f"{spk}_{wav_name.replace('.wav','.npy')}"
    mel_path = os.path.join(OUT_MELS,mel_file)
    np.save(mel_path,mel)

    clean_text = normalize_text(row["text"])

    rows_out.append([mel_path,clean_text,spk2id[spk]])

#Save mel data

pd.DataFrame(rows_out).to_csv(
    OUT_META,
    sep = "|",
    index = False,
    header = False 
)

print(f"\n Created {len(rows_out)} mel files.")
print(f"\n mel_metadata.csv")

