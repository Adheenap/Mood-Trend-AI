import sys
import os
import pandas as pd

# --------------------------------------------------
# Fix Python path (IMPORTANT)
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from models.text_emotion_model import predict_emotion

# --------------------------------------------------
# Simulated daily user text inputs
# --------------------------------------------------
daily_texts = [
    "I will kill her",
    "Work was stressful and exhausting",
    "Nothing special happened today",
    "I am worried about my future",
    "I feel tired and demotivated",
    "I am frustrated with my progress",
    "Today was a good and productive day"
]

# --------------------------------------------------
# Run inference
# --------------------------------------------------
results = []

for day, text in enumerate(daily_texts, start=1):
    prediction = predict_emotion(text)

    results.append({
        "day": day,
        "text": text,
        "dominant_emotion": prediction["dominant_emotion"],
        "mental_state": prediction["mental_state"],
        **prediction["emotion_percentages"]
    })

# --------------------------------------------------
# Save results
# --------------------------------------------------
df = pd.DataFrame(results)

output_path = "datasets/daily_text_emotions.csv"
df.to_csv(output_path, index=False)

print(" Text emotion inference completed")
print(" Saved to:", output_path)
print("\nPreview:")
print(df)
