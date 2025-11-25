import json
import torch 
import numpy as np 
import pandas as pd
from models.fastspeech2 import FastSpeech2

def test_acoustic_forward():
    # Load config
    cfg = json.load(open("configs/acoustic_config.json"))

    # Load metadata
    meta = pd.read_csv("data/processed/mel_metadata.csv", 
                       sep = "|" , 
                       names= ["mel", "text", "speaker"])

    row = meta.iloc[0]
    mel = np.load(row["mel"])
    text = row["text"]
    spk = int(row["speaker"])

    tokens = torch.tensor([ord(c) % cfg["vocab_size"] for c in text]).unsqueeze(0)
    spk = torch.tensor([spk])

    model = FastSpeech2(cfg, meta["speaker"].nunique()).eval()

    with torch.no_grad():
        mel_pred, dur = model(tokens, spk)
    
    assert mel_pred.shape[-1] == cfg["num_mels"]
    assert mel_pred.shape[0]  == 1
    assert dur.shape[0] == 1

    print("Phase 3 forward pass OK")
