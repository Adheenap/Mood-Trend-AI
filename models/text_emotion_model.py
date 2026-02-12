from transformers import pipeline

# --------------------------------------------------
# Load emotion classification model (PyTorch backend)
# --------------------------------------------------
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None,
    framework="pt"
)

# --------------------------------------------------
# Emotion → Mental State Mapping
# --------------------------------------------------
MENTAL_STATE_RULES = {
    "joy": "Healthy",
    "surprise": "Alert",
    "neutral": "Stable",
    "sadness": "Low Mood Risk",
    "fear": "Anxiety Risk",
    "anger": "Stress Risk",
    "disgust": "Emotional Discomfort"
}

# --------------------------------------------------
# Core Prediction (FULL OUTPUT)
# --------------------------------------------------
def predict_emotion(text):
    """
    Returns:
    - emotion_percentages
    - dominant_emotion
    - mental_state
    """

    results = classifier(text)[0]
    total_score = sum(r["score"] for r in results)

    emotion_percentages = {
        r["label"]: round((r["score"] / total_score) * 100, 2)
        for r in results
    }

    dominant_emotion = max(emotion_percentages, key=emotion_percentages.get)
    mental_state = MENTAL_STATE_RULES.get(dominant_emotion, "Unknown")

    return {
        "emotion_percentages": emotion_percentages,
        "dominant_emotion": dominant_emotion,
        "mental_state": mental_state
    }

# --------------------------------------------------
# BACKEND COMPATIBILITY FUNCTION (IMPORTANT)
# --------------------------------------------------
def predict_emotion_percentages(text):
    """
    Used by Flask backend
    Returns ONLY emotion percentages
    """
    return predict_emotion(text)["emotion_percentages"]

# --------------------------------------------------
# Test Run
# --------------------------------------------------
if __name__ == "__main__":
    sample = "I feel anxious and stressed about my work"
    print(predict_emotion(sample))


