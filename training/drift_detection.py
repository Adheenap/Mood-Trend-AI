import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress TF noise

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Load emotion trend data
# --------------------------------------------------
DATA_PATH = "datasets/emotion_trends.csv"
MODEL_PATH = "models/emotion_drift_lstm.h5"
OUTPUT_CSV = "datasets/drift_scores.csv"
PLOT_PATH = "datasets/emotion_drift_plot.png"
TEXT_OUTPUT = "datasets/mental_state_report.txt"

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Drift score calculation
# --------------------------------------------------
df["drift_score"] = abs(df["emotion_score"] - df["rolling_mean"])
df.to_csv(OUTPUT_CSV, index=False)

# --------------------------------------------------
# Mental state assessment
# --------------------------------------------------
avg_drift = df["drift_score"].mean()

if avg_drift < 0.3:
    state = "Stable emotional state"
    suggestion = "Emotional patterns are stable. Maintain healthy routines."
elif avg_drift < 0.7:
    state = "Moderate emotional fluctuation"
    suggestion = "Mild stress signals. Consider rest and self-care."
else:
    state = "High emotional instability"
    suggestion = "Strong emotional drift detected. Seek support and reduce stressors."

# --------------------------------------------------
# FORCE PRINT OUTPUT (flush)
# --------------------------------------------------
print("\n Mental State Assessment", flush=True)
print("State:", state, flush=True)
print("Suggestion:", suggestion, flush=True)

# --------------------------------------------------
# SAVE ADVICE TO FILE (IMPORTANT PROOF)
# --------------------------------------------------
with open(TEXT_OUTPUT, "w", encoding="utf-8") as f:
    f.write("Mental State Assessment\n")
    f.write(f"State: {state}\n")
    f.write(f"Suggestion: {suggestion}\n")

# --------------------------------------------------
# Plot emotion drift
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["day"], df["drift_score"], marker="o", label="Emotion Drift")
plt.axhline(avg_drift, color="red", linestyle="--", label="Average Drift")
plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.title("Emotion Drift Over Time")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_PATH)
plt.close()

print("\n Drift scores saved to:", OUTPUT_CSV, flush=True)
print(" Drift plot saved to:", PLOT_PATH, flush=True)
print(" Mental state report saved to:", TEXT_OUTPUT, flush=True)
