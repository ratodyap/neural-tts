import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from models.fastspeech2 import FastSpeech2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# make sure checkpoints folder exists
os.makedirs("checkpoints", exist_ok=True)

# load config
CONFIG = json.load(open("configs/acoustic_config_demo.json"))

# load metadata
META = pd.read_csv("data/processed/mel_metadata.csv",
                   sep="|",
                   names=["mel", "text", "speaker"])

# speaker remap
unique_speakers = sorted(META["speaker"].astype(int).unique())
speaker_to_index = {spk: i for i, spk in enumerate(unique_speakers)}

def tokenize(text):
    return torch.tensor([ord(c) % CONFIG["vocab_size"] for c in text],
                        dtype=torch.long)

def load_mel(path):
    return torch.tensor(np.load(path), dtype=torch.float32)

# dataset
class TTSDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = META

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_path, text, spk = self.data.iloc[idx]
        t = tokenize(text)
        mel = load_mel(mel_path)
        spk = speaker_to_index[int(spk)]
        return t, mel, spk

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

# print basic info
# print training device, dataset size, batch size and epochs
print("Training on:", DEVICE)
print("Dataset size:", len(dataset))
print("Batch size:", CONFIG["batch_size"])
print("Epochs:", CONFIG["epochs"])

for epoch in range(CONFIG["epochs"]):

    # print start of epoch
    print("Starting epoch:", epoch+1)

    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):

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

        # align lengths
        pred_len = mel_pred.shape[1]
        true_len = mels_padded.shape[1]

        if pred_len > true_len:
            mel_pred = mel_pred[:, :true_len, :]
        elif pred_len < true_len:
            pad = true_len - pred_len
            mel_pred = torch.nn.functional.pad(
                mel_pred, (0,0,0,pad), mode="constant", value=0.0
            )

        loss = criterion(mel_pred, mels_padded)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # print loss every 5 batches
        # this helps confirm training is running
        if (batch_idx + 1) % 5 == 0:
            print("Epoch", epoch+1, "Batch", batch_idx+1, "Loss:", loss.item())

    # print epoch completion
    print("Finished epoch:", epoch+1, "Total Loss:", total_loss)

    # save checkpoint
    ckpt_path = f"checkpoints/acoustic_demo_epoch{epoch+1}.pth"
    torch.save(model.state_dict(), ckpt_path)

    # print checkpoint saved info
    print("Saved checkpoint at:", ckpt_path)

print("Training complete.")
