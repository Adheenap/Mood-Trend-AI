import pandas as pd
import os

# --------------------------------------------------
# 1. Load daily text emotion predictions
# --------------------------------------------------
input_file = "datasets/daily_text_emotions.csv"

if not os.path.exists(input_file):
    raise FileNotFoundError("❌ daily_text_emotions.csv not found. Run run_text_inference.py first.")

df = pd.read_csv(input_file)
print("Loaded daily emotion data:")
print(df.head())

# --------------------------------------------------
# 2. Encode mental states numerically
# --------------------------------------------------
# positive = +1, neutral = 0, negative = -1
emotion_encoding = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

df["emotion_score"] = df["mental_state"].map(emotion_encoding)

# --------------------------------------------------
# 3. Build rolling emotion trend features
# --------------------------------------------------
window_size = 3  # rolling window (days)

df["rolling_mean"] = df["emotion_score"].rolling(window=window_size).mean()
df["rolling_std"] = df["emotion_score"].rolling(window=window_size).std()
df["rolling_min"] = df["emotion_score"].rolling(window=window_size).min()
df["rolling_max"] = df["emotion_score"].rolling(window=window_size).max()

# Fill NaN values from rolling window
df = df.fillna(0)

# --------------------------------------------------
# 4. Save trend features
# --------------------------------------------------
output_file = "datasets/emotion_trends.csv"
df.to_csv(output_file, index=False)

print("\n Emotion trends built successfully")
print(" Saved to:", output_file)
print("\nPreview:")
print(df)

