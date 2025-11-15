import os
import librosa
import soundfile as sf
import tqdm

src_root = "data/raw/VCTK/wav48"
dst_root = "data/processed"
os.makedirs(dst_root, exist_ok=True)

for spk in tqdm.tqdm(os.listdir(src_root)):
    spk_src = os.path.join(src_root, spk)
    spk_dst = os.path.join(dst_root, spk)

    # Skip non-directories like log.txt
    if not os.path.isdir(spk_src):
        continue

    os.makedirs(spk_dst, exist_ok=True)

    for file in os.listdir(spk_src):
        fname = file.lower()
        
         # Process .wav and .flac only
        if not (fname.endswith("_mic1.flac") or (fname.endswith("_mic1.wav"))):
            continue

        in_path = os.path.join(spk_src, file)

        # remove "_mic1" and set output.wav
        base = os.path.splitext(file)[0].replace("_mic1","")
        out_path = os.path.join(spk_dst, base + ".wav")

        # Load audio
        y, _ = librosa.load(in_path, sr=22050, mono=True)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Write as .wav
        sf.write(out_path, y, 22050)
