import pandas as pd

# --------------------------------------------------
# Load multimodal daily logs
# --------------------------------------------------
INPUT_PATH = "backend/datasets/multimodal_daily_logs.csv"
OUTPUT_PATH = "datasets/multimodal_emotion_trends.csv"

df = pd.read_csv(INPUT_PATH)

print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

# --------------------------------------------------
# Ensure numeric fusion_score
# --------------------------------------------------
df["fusion_score"] = pd.to_numeric(df["fusion_score"], errors="coerce")
df = df.dropna(subset=["fusion_score"]).reset_index(drop=True)

# --------------------------------------------------
# Add day index
# --------------------------------------------------
df["day"] = range(1, len(df) + 1)

# --------------------------------------------------
# Rolling emotion statistics
# --------------------------------------------------
WINDOW = 3

df["rolling_mean"] = df["fusion_score"].rolling(WINDOW, min_periods=1).mean()
df["rolling_std"] = df["fusion_score"].rolling(WINDOW, min_periods=1).std().fillna(0)
df["rolling_min"] = df["fusion_score"].rolling(WINDOW, min_periods=1).min()
df["rolling_max"] = df["fusion_score"].rolling(WINDOW, min_periods=1).max()

# --------------------------------------------------
# Save trends
# --------------------------------------------------
df.to_csv(OUTPUT_PATH, index=False)

print("✅ Multimodal emotion trends created")
print("📁 Saved to:", OUTPUT_PATH)
print("\nPreview:")
print(df.head())
