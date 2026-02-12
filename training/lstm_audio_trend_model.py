import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Load processed audio trend data
# --------------------------------------------------
df = pd.read_csv("datasets/audio_emotion_trends_processed.csv")

features = [
    "emotion_score",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max"
]

data = df[features].values
print("Loaded audio trend data shape:", data.shape)

# --------------------------------------------------
# Normalize
# --------------------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------------------------------
# Create sequences (FIXED)
# --------------------------------------------------
SEQ_LEN = 3   # ✅ reduced for small dataset

X, y = [], []

for i in range(len(data_scaled) - SEQ_LEN):
    X.append(data_scaled[i:i + SEQ_LEN])
    y.append(data_scaled[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# --------------------------------------------------
# Safety check
# --------------------------------------------------
if len(X) == 0:
    raise ValueError(" Not enough audio data to train LSTM")

# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, X.shape[2])),
    LSTM(64),
    Dense(X.shape[2])
])

model.compile(optimizer="adam", loss="mse")
model.summary()

# --------------------------------------------------
# Train
# --------------------------------------------------
early_stop = EarlyStopping(patience=3, restore_best_weights=True)

model.fit(
    X,
    y,
    epochs=20,
    batch_size=4,
    callbacks=[early_stop]
)

# --------------------------------------------------
# Save model
# --------------------------------------------------
model.save("models/audio_drift_lstm.h5")

print(" Audio LSTM drift model saved at models/audio_drift_lstm.h5")

