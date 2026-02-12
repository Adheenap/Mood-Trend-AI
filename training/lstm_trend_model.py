import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Load emotion trend data
# --------------------------------------------------
data_path = "datasets/emotion_trends.csv"
df = pd.read_csv(data_path)

print("Loaded data shape:", df.shape)
print("Columns:", list(df.columns))

# --------------------------------------------------
# Use signal-based emotion features
# --------------------------------------------------
feature_cols = [
    "emotion_score",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max"
]

data = df[feature_cols].values

# --------------------------------------------------
# Normalize features (0–1)
# --------------------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------------------------------
# Create sequences for LSTM
# --------------------------------------------------
SEQUENCE_LENGTH = 3   # small because dataset is small

X, y = [], []

for i in range(len(data_scaled) - SEQUENCE_LENGTH):
    X.append(data_scaled[i:i + SEQUENCE_LENGTH])
    y.append(data_scaled[i + SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("❌ Not enough data to train LSTM")

# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------
model = Sequential([
    Input(shape=(SEQUENCE_LENGTH, len(feature_cols))),
    LSTM(32),
    Dense(len(feature_cols))
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# --------------------------------------------------
# Train model
# --------------------------------------------------
early_stop = EarlyStopping(
    monitor="loss",
    patience=3,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=100,
    batch_size=2,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------------------------------
# Save trained model
# --------------------------------------------------
model.save("models/emotion_drift_lstm.h5")

print(" LSTM drift model saved at models/emotion_drift_lstm.h5")
