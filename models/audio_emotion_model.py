# models/audio_emotion_model.py

import librosa
import numpy as np

# --------------------------------------------------
# Extract basic audio features
# --------------------------------------------------
def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)

    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))

    return rms, zcr, mfcc


# --------------------------------------------------
# Estimate emotion percentages (heuristic-based)
# --------------------------------------------------
def predict_voice_emotion_percentages(audio_path):
    rms, zcr, mfcc = extract_audio_features(audio_path)

    emotions = {
        "calm": max(0, 1 - rms * 3),
        "angry": min(1, rms * 2),
        "fear": min(1, zcr * 3),
        "sad": max(0, 1 - mfcc / 100),
        "happy": min(1, mfcc / 80)
    }

    total = sum(emotions.values())

    percentages = {
        k: round((v / total) * 100, 2)
        for k, v in emotions.items()
    }

    return percentages


