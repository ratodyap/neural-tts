import os
import glob
import shutil

MEL_DIR = "data/processed/mels"
SPK_BASE = "data/processed"
OUT_DIR = "data/processed/wavs"

os.makedirs(OUT_DIR, exist_ok=True)

mel_files = glob.glob(f"{MEL_DIR}/*.npy")
missing = []

for mel in mel_files:
    mel_name = os.path.basename(mel)           # p225_p225_001.npy
    parts = mel_name.split("_")                # ['p225','p225','001.npy']

    spk = parts[0]                             # 'p225'
    idx = parts[2].replace(".npy", "")         # '001'

    wav_name = f"{spk}_{idx}.wav"              # p225_001.wav

    src = os.path.join(SPK_BASE, spk, wav_name)
    dst = os.path.join(OUT_DIR, wav_name)

    if os.path.exists(src):
        shutil.copyfile(src, dst)
    else:
        missing.append(src)

print("Copied:", len(mel_files) - len(missing))
print("Missing:", len(missing))
