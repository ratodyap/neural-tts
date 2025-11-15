import os, csv, json

root = "data/raw/VCTK/wav48"
text_root = "data/raw/VCTK/txt"
metadata = []

speakers = sorted([s for s in os.listdir(root) if os.path.isdir(os.path.join(root, s))])

for spk in speakers:
    wav_dir, txt_dir = os.path.join(root, spk), os.path.join(text_root, spk)
    
    for file in os.listdir(wav_dir):
        
        # Accept only mic1 files (.flac or .wav)
        fname = file.lower()
        if not (fname.endswith("_mic1.flac")) or fname.endswith(("_mic1.wav")):
            continue

        # remove "_mic1"
        base = os.path.splitext(file)[0].replace("_mic1","")

        txt_path = os.path.join(txt_dir, base + ".txt")

        if not os.path.exists(txt_path):
            continue

        text = open(txt_path, encoding = "utf-8").read().strip()

        # Convert audio  to processed/<spk>/<base>.wav later
        metadata.append([f"{spk}/{base}.wav", text, spk])

os.makedirs("data", exist_ok=True)
with open("data/metadata.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f, delimiter="|").writerows(metadata)

spk2id = {spk: i for i, spk in enumerate(speakers)}
json.dump(spk2id, open("data/spk2id.json", "w"))

print(f"✅ Found {len(metadata)} audio-text pairs and saved metadata.csv")
