import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load drift data
# --------------------------------------------------
df = pd.read_csv("datasets/drift_scores.csv")

# --------------------------------------------------
# Plot emotion drift
# --------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df["day"], df["predicted_drift"], marker="o", label="Emotion Drift")
plt.axhline(y=0, color="green", linestyle="--", label="Stable Line")
plt.axhline(y=-0.2, color="red", linestyle="--", label="Risk Threshold")

plt.xlabel("Day")
plt.ylabel("Drift Score")
plt.title("Emotion Drift Over Time")
plt.legend()
plt.grid(True)

plt.show()

# --------------------------------------------------
# Mental condition suggestion
# --------------------------------------------------
negative_days = df[df["predicted_drift"] < -0.2]

if len(negative_days) >= 3:
    print("\n Mental Condition Suggestion:")
    print("Sustained negative emotional drift detected.")
    print("Possible risk of stress / burnout.")
    print("Recommendation: Emotional check-in or counseling.")
else:
    print("\n Mental Condition Suggestion:")
    print("Emotional state appears stable.")

