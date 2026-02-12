import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("datasets/audio_emotion_trends_processed.csv")

features = [
    "emotion_score",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max"
]

data = df[features].values
print("Loaded rows:", len(df))

# --------------------------------------------------
# Normalize
# --------------------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------------------------------
# Load model
# --------------------------------------------------
model = load_model("models/audio_drift_lstm.h5", compile=False)

# --------------------------------------------------
# Drift detection
# --------------------------------------------------
SEQ_LEN = 3  # ✅ MUST MATCH training
drift_scores = []

for i in range(len(data_scaled) - SEQ_LEN):
    seq = data_scaled[i:i + SEQ_LEN]
    pred = model.predict(seq.reshape(1, SEQ_LEN, -1), verbose=0)
    error = np.mean(np.abs(pred - data_scaled[i + SEQ_LEN]))
    drift_scores.append(error)

# --------------------------------------------------
# Attach drift scores
# --------------------------------------------------
if len(drift_scores) == 0:
    raise ValueError("❌ Not enough audio data to compute drift")

df = df.iloc[SEQ_LEN:].copy()
df["drift_score"] = drift_scores

# --------------------------------------------------
# Plot (FIXED)
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(df.index + 1, df["drift_score"], marker="o", linewidth=2)
plt.title("Audio Emotion Drift Over Time")
plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("datasets/audio_drift_plot.png")
plt.show()

# --------------------------------------------------
# Mental state assessment
# --------------------------------------------------
avg_drift = df["drift_score"].mean()

if avg_drift < 0.05:
    state = "Stable emotional state"
    advice = "Emotional balance is good. Maintain routine."
elif avg_drift < 0.1:
    state = "Moderate emotional fluctuation"
    advice = "Mild stress signals. Take breaks and relax."
else:
    state = "High emotional instability"
    advice = "Strong stress indicators. Consider professional help."

print("\n🎧 Audio Mental State Assessment")
print("State:", state)
print("Suggestion:", advice)

# --------------------------------------------------
# Save results
# --------------------------------------------------
df.to_csv("datasets/audio_drift_scores.csv", index=False)
print("\n Audio drift scores saved to datasets/audio_drift_scores.csv")
