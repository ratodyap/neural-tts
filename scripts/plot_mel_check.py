import numpy as np 
import matplotlib.pyplot as plt 
import glob, random 


mel_files = glob.glob("data/processed/mels/*.npy")
mel = np.load(random.choice(mel_files))

plt.imshow(mel.T, origin="lower", aspect="auto")
plt.title("Sample Mel Spectrogram")
plt.xlabel("Time Frames")
plt.ylabel("Mel Bins")
plt.show()