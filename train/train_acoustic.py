import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from models.fastspeech2 import FastSpeech2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# correct config file
CONFIG = json.load(open("configs/acoustic_config.json"))

# correct metadata column names
META = pd.read_csv("data/processed/mel_metadata.csv",
                   sep="|",
                   names=["mel", "text", "speaker"])

def tokenize(text):
    return torch.tensor([ord(c) % CONFIG["vocab_size"] for c in text],
                        dtype=torch.long)

def load_mel(path):
    return torch.tensor(np.load(path), dtype=torch.float32)

# Dataset
class TTSDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = META

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_path, text, spk = self.data.iloc[idx]
        t = tokenize(text)
        mel = load_mel(mel_path)
        return t, mel, int(spk)


dataset = TTSDataset()
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    collate_fn=lambda batch: batch
)

num_speakers = META["speaker"].nunique()
model = FastSpeech2(CONFIG, num_speakers).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
criterion = nn.MSELoss()

print("Training on:", DEVICE)

for epoch in range(CONFIG["epochs"]):
    total_loss = 0.0

    for batch in loader:
        texts, mels, spks = zip(*batch)

        texts = nn.utils.rnn.pad_sequence(texts, batch_first=True).to(DEVICE)
        spks = torch.tensor(spks).to(DEVICE)

        max_len = max(m.shape[0] for m in mels)
        mels_padded = torch.zeros(len(mels), max_len, CONFIG["num_mels"])

        for i, m in enumerate(mels):
            mels_padded[i, :m.shape[0]] = m

        mels_padded = mels_padded.to(DEVICE)

        optimizer.zero_grad()

        mel_pred, durations = model(texts, spks)
        mel_pred = mel_pred[:, :mels_padded.shape[1], :]

        loss = criterion(mel_pred, mels_padded)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "checkpoints/acoustic_model.pth")
print("Acoustic model saved.")
