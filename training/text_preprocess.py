import pandas as pd
import kagglehub
import os

# --------------------------------------------------
# 1. Load GoEmotions dataset
# --------------------------------------------------
go_path = kagglehub.dataset_download("debarshichanda/goemotions")
print("GoEmotions root path:", go_path)

train_file = os.path.join(go_path, "data", "train.tsv")
labels_file = os.path.join(go_path, "data", "emotions.txt")

# --------------------------------------------------
# 2. Load emotion label names
# --------------------------------------------------
with open(labels_file, "r") as f:
    emotion_labels = [line.strip() for line in f.readlines()]

print("Total emotion labels:", len(emotion_labels))

# --------------------------------------------------
# 3. Load training data
# --------------------------------------------------
df = pd.read_csv(train_file, sep="\t", header=None)
df.columns = ["text", "label_ids", "id"]
print("Original dataset size:", df.shape)

# --------------------------------------------------
# 4. Handle MULTI-LABEL properly
# --------------------------------------------------
# Take first label if multiple labels exist
def get_primary_label(label):
    if "," in str(label):
        return int(label.split(",")[0])
    return int(label)

df["primary_label"] = df["label_ids"].apply(get_primary_label)
df["emotion_name"] = df["primary_label"].apply(lambda x: emotion_labels[x])

# --------------------------------------------------
# 5. Map fine emotions → mental states
# --------------------------------------------------
positive_emotions = {
    "joy", "love", "gratitude", "optimism", "pride",
    "relief", "approval", "excitement", "amusement"
}

negative_emotions = {
    "anger", "sadness", "fear", "disgust", "disappointment",
    "grief", "annoyance", "embarrassment", "nervousness",
    "remorse"
}

def map_to_state(emotion):
    if emotion in positive_emotions:
        return "positive"
    elif emotion in negative_emotions:
        return "negative"
    else:
        return "neutral"

df["emotion"] = df["emotion_name"].apply(map_to_state)

# --------------------------------------------------
# 6. Clean and save
# --------------------------------------------------
df = df[df["text"].str.len() > 5]

final_df = df[["text", "emotion"]]

output_path = "datasets/text_emotions_clean.csv"
final_df.to_csv(output_path, index=False)

print(" Clean dataset saved to:", output_path)
print("\nEmotion distribution:")
print(final_df["emotion"].value_counts())
