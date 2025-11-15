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

    for wav in os.listdir(spk_src):

        # Process .wav and .flac only
        if not wav.lower().endswith((".wav", ".flac")):
            continue

        src_path = os.path.join(spk_src, wav)

        # Load audio
        y, _ = librosa.load(src_path, sr=22050, mono=True)

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)

        # Save output as .wav
        out_name = os.path.splitext(wav)[0] + ".wav"
        out_path = os.path.join(spk_dst, out_name)

        sf.write(out_path, y, 22050)
