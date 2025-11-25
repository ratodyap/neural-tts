import os
import json
import numpy as np
import pandas as pd
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from scipy.io.wavfile import write
from models.hifigan import HiFiGANGenerator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open("configs/vocoder_config_demo.json"))

meta = pd.read_csv(
    "data/processed/mel_metadata.csv",
    sep="|",
    names=["mel", "text", "speaker"]
)

def load_mel(path):
    return torch.tensor(np.load(path), dtype=torch.float32)


# ---------------------------------------------------
# Dataset
# ---------------------------------------------------

class VocDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(meta)

    def __getitem__(self, idx):
        mel_path, _, _ = meta.iloc[idx]
        mel = load_mel(mel_path).T  # (freq, time)

        mel_name = os.path.basename(mel_path)
        parts = mel_name.split("_")  # p225_p225_0001.npy

        spk = parts[0]
        idx_str = parts[2].replace(".npy", "")
        wav_name = f"{spk}_{idx_str}.wav"

        wav_path = os.path.join("data/processed", spk, wav_name)

        if not os.path.exists(wav_path):
            raise FileNotFoundError("Missing WAV: " + wav_path)

        wav, sr = torchaudio.load(wav_path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        return mel, wav


def collate_fn(batch):
    return batch


dataset = VocDataset()

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=cfg["batch_size"],
    shuffle=True,
    collate_fn=collate_fn
)


# ---------------------------------------------------
# Model
# ---------------------------------------------------

model = HiFiGANGenerator(cfg).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"])
criterion = nn.L1Loss()

os.makedirs("checkpoints/hifigan", exist_ok=True)
os.makedirs("validation_audio", exist_ok=True)

print("Training HiFi-GAN on:", DEVICE)


# ---------------------------------------------------
# Training Loop
# ---------------------------------------------------

for epoch in range(1, cfg["epochs"] + 1):
    total_loss = 0.0
    print("\nStarting Epoch:", epoch)

    for batch_idx, batch in enumerate(loader):
        
        # For small demo need to comment it out when it is finally done
        if batch_idx > 200:
            break
        mels, wavs = zip(*batch)

        # ---- pad mel (pad time axis) ----
        max_mel_len = max(m.shape[1] for m in mels)
        mel_list = []
        for m in mels:
            if m.shape[1] < max_mel_len:
                pad = max_mel_len - m.shape[1]
                m = torch.nn.functional.pad(m, (0, pad, 0, 0))
            mel_list.append(m)
        mel = torch.stack(mel_list).to(DEVICE)

        # ---- pad wav (pad time axis) ----
        max_wav_len = max(w.shape[-1] for w in wavs)
        wav_list = []
        for w in wavs:
            if w.shape[-1] < max_wav_len:
                pad = max_wav_len - w.shape[-1]
                w = torch.nn.functional.pad(w, (0, pad))
            wav_list.append(w)
        wav = torch.stack(wav_list).to(DEVICE)

        # ---------------------------------------------------
        # Forward pass
        # ---------------------------------------------------
        pred = model(mel)

        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)

        # ---------------------------------------------------
        # Trim to equal length BEFORE loss
        # ---------------------------------------------------
        pred_len = pred.shape[-1]
        wav_len = wav.shape[-1]
        min_len = min(pred_len, wav_len)

        pred = pred[..., :min_len]
        wav = wav[..., :min_len]

        pred = pred.squeeze(1)
        wav = wav.squeeze(1)

        # ---------------------------------------------------
        # Compute loss
        # ---------------------------------------------------
        loss = criterion(pred, wav)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---------------------------------------------------
        # Progress printing (similar to train_acoustic.py)
        # ---------------------------------------------------
        print(
            "Epoch", epoch,
            "| Batch", batch_idx + 1, "/", len(loader),
            "| Mel shape:", mel.shape,
            "| Wav shape:", wav.shape,
            "| Pred shape:", pred.shape,
            "| Loss:", loss.item()
        )

    # ---------------------------------------------------
    # End of epoch: print average loss & save checkpoint
    # ---------------------------------------------------
    avg_loss = total_loss / len(loader)
    print("Epoch", epoch, "completed. Average Loss:", avg_loss)

    ckpt_path = f"checkpoints/hifigan/epoch_{epoch}.pth"
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint:", ckpt_path)

    # save validation audio
    with torch.no_grad():
        audio = pred[0].cpu().numpy()
        write(f"validation_audio/epoch_{epoch}.wav", 22050, audio)

print("Training complete.")
