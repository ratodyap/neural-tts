import torch
import torch.nn as nn 

class FastSpeech2(nn.Module):

    def __init__(self, config, num_speakers):
        super().__init__()
        self.hidden = config["hidden_size"] 

        # Text embedding 
        self.embed  = nn.Embedding(config["vocab_size"], self.hidden)

        # Speaker embedding + Projection to hidden 
        self.spk_embed = nn.Embedding(num_speakers,config["speaker_emb_dim"])
        self.spk_proj = nn.Linear(config["speaker_emb_dim"], self.hidden)

        #encoder

        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.hidden,
            nhead = config["num_heads"],
            dim_feedforward= self.hidden * 4,
            dropout= config["dropout"],
            batch_first= True
        ) 

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers= config["num_layers"]
        )
       
        # duration predictor 
        self.duration_pred = nn.Linear(self.hidden, 1)

        # decoder -> mel generator
        self.mel_out = nn.Linear(self.hidden, config["num_mels"])

    def forward(self, text_ids, speaker_ids):
        x = self.embed(text_ids)

        spk = self.spk_embed(speaker_ids)
        spk = self.spk_proj(spk)
        spk = spk.unsqueeze(1)

        x = x + spk

        x = self.encoder(x)

        durations = self.duration_pred(x).squeeze(-1)

        mel = self.mel_out(x)
        return mel, durations