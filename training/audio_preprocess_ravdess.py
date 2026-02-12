import os
import librosa
import numpy as np
import pandas as pd

RAVDESS_PATH = r"C:\Users\...\ravdess-emotional-speech-audio"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

features = []

for root, _, files in os.walk(RAVDESS_PATH):
    for file in files:
        if file.endswith(".wav"):
            emotion_id = file.split("-")[2]
            emotion = emotion_map.get(emotion_id)

            file_path = os.path.join(root, file)
            y, sr = librosa.load(file_path, sr=None)

            mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            energy = np.mean(librosa.feature.rms(y=y))

            features.append(
                list(mfcc) + [zcr, energy, emotion]
            )

columns = [f"mfcc_{i}" for i in range(13)] + ["zcr", "energy", "emotion"]
df = pd.DataFrame(features, columns=columns)

df.to_csv("datasets/audio_training_features.csv", index=False)
print("Audio training features saved")
