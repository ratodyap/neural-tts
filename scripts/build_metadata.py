import os, csv, json

root = "data/raw/VCTK/wav48"
text_root = "data/raw/VCTK/txt"
metadata = []

speakers = sorted([s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))])

for spk in speakers:
    wav_dir, txt_dir = os.path.join(root, spk), os.path.join(text_root, spk)
    for wav in os.listdir(wav_dir):
        # Only accept .wav or .flac and only mic1
        if not wav.lower().endswith((".wav", ".flac")) or "_mic2" in wav:
            continue

        # Remove _mic1/_mic2 suffix for matching text
        base = os.path.splitext(wav)[0].replace("_mic1", "").replace("_mic2", "")
        txt_path = os.path.join(txt_dir, base + ".txt")

        if os.path.exists(txt_path):
            with open(txt_path, encoding="utf-8") as f:
                text = f.read().strip()
            metadata.append([f"{spk}/{wav}", text, spk])

os.makedirs("data", exist_ok=True)
with open("data/metadata.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f, delimiter="|").writerows(metadata)

spk2id = {spk: i for i, spk in enumerate(speakers)}
json.dump(spk2id, open("data/spk2id.json", "w"))

#print(f"✅ Found {len(metadata)} audio-text pairs and saved metadata.csv")
