import random, librosa, librosa.display
import matplotlib.pyplot as plt
import os

spk = random.choice(os.listdir("data/processed"))
file = random.choice(os.listdir(f"data/processed/{spk}"))
y, sr = librosa.load(f"data/processed/{spk}/{file}", sr=22050)

plt.figure(figsize=(10, 3))
librosa.display.waveshow(y, sr=sr)   # SAFE VERSION
plt.title(f"{spk}/{file}")
plt.tight_layout()
plt.show()
