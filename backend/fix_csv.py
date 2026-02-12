import pandas as pd

PATH = "datasets/multimodal_daily_logs.csv"

df = pd.read_csv(PATH, engine="python")

# Ensure all required columns exist
for col in ["audio_score"]:
    if col not in df.columns:
        df[col] = None

# Reorder columns (IMPORTANT)
df = df[[
    "date",
    "text_score",
    "audio_score",
    "fusion_score",
    "dominant_emotion",
    "mental_state"
]]

df.to_csv(PATH, index=False)
print(" CSV fixed successfully")
