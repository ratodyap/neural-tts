import  json
import torch
from models.hifigan import HiFiGANGenerator

def test_vocoder_forward():
    cfg = json.load(open("configs/vocoder_config_demo.json"))
    model = HiFiGANGenerator(cfg).eval()

    mel = torch.rand(1, cfg["num_mels"],50)

    with torch.no_grad():
        audio = model(mel)

    assert audio.ndim ==3
    assert audio.shape[1] ==1
    print("Vocoder forward pas OK") 