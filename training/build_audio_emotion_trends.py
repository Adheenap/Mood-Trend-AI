import pandas as pd

# --------------------------------------------------
# Load audio emotion inference output
# --------------------------------------------------
input_path = "datasets/audio_emotion_trends.csv"
df = pd.read_csv(input_path)

print("Loaded audio emotion data:")
print(df.head())

# --------------------------------------------------
# Define emotion polarity
# --------------------------------------------------
positive = ["happy", "surprise"]
negative = ["sad", "angry", "fear", "disgust"]

# --------------------------------------------------
# Compute emotion score per day
# --------------------------------------------------
def compute_score(row):
    score = 0
    for emo in positive:
        if emo in row:
            score += row[emo]
    for emo in negative:
        if emo in row:
            score -= row[emo]
    return round(score / 100, 3)

df["emotion_score"] = df.apply(compute_score, axis=1)

# --------------------------------------------------
# Rolling statistics (trend features)
# --------------------------------------------------
df["rolling_mean"] = df["emotion_score"].rolling(window=3, min_periods=1).mean()
df["rolling_std"] = df["emotion_score"].rolling(window=3, min_periods=1).std().fillna(0)
df["rolling_min"] = df["emotion_score"].rolling(window=3, min_periods=1).min()
df["rolling_max"] = df["emotion_score"].rolling(window=3, min_periods=1).max()

# --------------------------------------------------
# Save processed trends
# --------------------------------------------------
output_path = "datasets/audio_emotion_trends_processed.csv"
df.to_csv(output_path, index=False)

print(" Audio emotion trends built")
print(" Saved to:", output_path)
print(df)

