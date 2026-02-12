import os
import librosa
import numpy as np
import pandas as pd
import kagglehub

print(" Starting audio preprocessing (RAVDESS)")

# --------------------------------------------------
# Download / Locate RAVDESS dataset
# --------------------------------------------------
dataset_path = kagglehub.dataset_download(
    "uwrfkaggler/ravdess-emotional-speech-audio"
)

print(" Dataset path:", dataset_path)

# --------------------------------------------------
# Emotion mapping from filename
# --------------------------------------------------
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

features = []

# --------------------------------------------------
# Walk through all audio files
# --------------------------------------------------
for root, _, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)

            try:
                # Load audio
                y, sr = librosa.load(file_path, duration=3, offset=0.5)

                # Extract features
                mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
                rms = np.mean(librosa.feature.rms(y=y))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y))

                # Get emotion from filename
                emotion_code = file.split("-")[2]
                emotion = emotion_map.get(emotion_code)

                if emotion is None:
                    continue

                feature_row = list(mfcc) + [rms, zcr, emotion]
                features.append(feature_row)

            except Exception as e:
                print("❌ Error processing:", file_path)
                print(e)

# --------------------------------------------------
# Create DataFrame
# --------------------------------------------------
columns = [f"mfcc_{i+1}" for i in range(13)] + ["rms", "zcr", "emotion"]
df = pd.DataFrame(features, columns=columns)

# --------------------------------------------------
# Save processed features
# --------------------------------------------------
output_path = "datasets/audio_training_features.csv"
df.to_csv(output_path, index=False)

print(" Audio preprocessing completed")
print(" Saved to:", output_path)
print(" Total samples:", df.shape[0])
print(df["emotion"].value_counts())
