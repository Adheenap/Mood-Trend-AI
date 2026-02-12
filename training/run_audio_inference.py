import os
import sys
import pandas as pd

# --------------------------------------------------
# Fix Python path to access project root
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.audio_emotion_model import predict_voice_emotion_percentages

# --------------------------------------------------
# Audio directsory (daily recordings)
# --------------------------------------------------
AUDIO_DIR = "datasets/sample_audio"

results = []

for i in range(1, 6):
    audio_path = f"{AUDIO_DIR}/day{i}.wav"

    if not os.path.exists(audio_path):
        print(f"⚠️ Missing audio file: {audio_path}")
        continue

    emotions = predict_voice_emotion_percentages(audio_path)

    results.append({
        "day": i,
        **emotions
    })

# --------------------------------------------------
# Save results
# --------------------------------------------------
df = pd.DataFrame(results)

output_path = "datasets/audio_emotion_trends.csv"
df.to_csv(output_path, index=False)

print(" Audio emotion inference completed")
print(" Saved to:", output_path)
print("\nPreview:")
print(df)

