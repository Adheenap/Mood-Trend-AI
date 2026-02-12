import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load multimodal emotion trends
# --------------------------------------------------
df = pd.read_csv("datasets/multimodal_emotion_trends.csv")

# --------------------------------------------------
# Plot fusion score and rolling mean
# --------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    df["day"],
    df["fusion_score"],
    marker="o",
    label="Daily Emotion Score"
)

plt.plot(
    df["day"],
    df["rolling_mean"],
    linestyle="--",
    linewidth=2,
    label="Emotional Baseline (Rolling Mean)"
)

plt.fill_between(
    df["day"],
    df["rolling_min"],
    df["rolling_max"],
    alpha=0.2,
    label="Emotional Range"
)

plt.title("Multimodal Emotion Drift Over Time")
plt.xlabel("Day")
plt.ylabel("Emotion Polarity Score")
plt.grid(True)
plt.legend()
plt.tight_layout()

# --------------------------------------------------
# Save + show
# --------------------------------------------------
plt.savefig("datasets/multimodal_emotion_drift.png")
#plt.show()

print("📊 Multimodal emotion drift graph saved at:")
print("datasets/multimodal_emotion_drift.png")

