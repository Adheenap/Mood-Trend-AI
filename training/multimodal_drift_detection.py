import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load drift data
# --------------------------------------------------
text_df = pd.read_csv("datasets/drift_scores.csv")
audio_df = pd.read_csv("datasets/audio_drift_scores.csv")

print("Text drift rows:", len(text_df))
print("Audio drift rows:", len(audio_df))

# --------------------------------------------------
# Align by day (safe merge)
# --------------------------------------------------
merged = pd.merge(
    text_df[["day", "drift_score"]],
    audio_df[["day", "drift_score"]],
    on="day",
    suffixes=("_text", "_audio")
)

print("Merged rows:", len(merged))

# --------------------------------------------------
# Weighted fusion
# --------------------------------------------------
TEXT_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4

merged["fusion_drift"] = (
    TEXT_WEIGHT * merged["drift_score_text"]
    + AUDIO_WEIGHT * merged["drift_score_audio"]
)

# --------------------------------------------------
# Plot fusion drift
# --------------------------------------------------
plt.figure(figsize=(9, 4))
plt.plot(
    merged["day"],
    merged["fusion_drift"],
    marker="o",
    linewidth=2,
    label="Fusion Drift"
)

plt.title("Multimodal Emotion Drift (Text + Audio)")
plt.xlabel("Day")
plt.ylabel("Fusion Drift Score")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("datasets/multimodal_drift_plot.png")
plt.show()

# --------------------------------------------------
# Mental health assessment
# --------------------------------------------------
avg_drift = merged["fusion_drift"].mean()

if avg_drift < 0.05:
    state = "Emotionally Stable"
    advice = "Healthy emotional balance. Maintain habits."
elif avg_drift < 0.1:
    state = "Mild Emotional Stress"
    advice = "Some stress detected. Rest and mindfulness suggested."
else:
    state = "High Emotional Risk"
    advice = "Strong distress indicators. Professional support advised."

print("\n FINAL MULTIMODAL MENTAL STATE")
print("State:", state)
print("Suggestion:", advice)

# --------------------------------------------------
# Save results
# --------------------------------------------------
merged.to_csv("datasets/multimodal_drift_scores.csv", index=False)
print("\n Multimodal drift saved to datasets/multimodal_drift_scores.csv")
print(" Plot saved as datasets/multimodal_drift_plot.png")
