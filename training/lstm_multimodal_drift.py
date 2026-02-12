import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# Load multimodal daily logs
# --------------------------------------------------
df = pd.read_csv("backend/datasets/multimodal_daily_logs.csv")

print("Loaded multimodal data shape:", df.shape)

# --------------------------------------------------
# Use fusion-based emotional signal
# --------------------------------------------------
features = ["fusion_score"]

data = df[features].values

# --------------------------------------------------
# Normalize (0–1)
# --------------------------------------------------
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# --------------------------------------------------
# Create sequences
# --------------------------------------------------
SEQ_LEN = 5   # small because data grows daily

X, y = [], []

for i in range(len(data_scaled) - SEQ_LEN):
    X.append(data_scaled[i:i + SEQ_LEN])
    y.append(data_scaled[i + SEQ_LEN])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

if len(X) == 0:
    raise ValueError("❌ Not enough multimodal data for LSTM training")

# --------------------------------------------------
# Build LSTM model
# --------------------------------------------------
model = Sequential([
    Input(shape=(SEQ_LEN, X.shape[2])),
    LSTM(32),
    Dense(1)
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
    patience=3,
    restore_best_weights=True
)

model.fit(
    X,
    y,
    epochs=50,
    batch_size=2,
    callbacks=[early_stop],
    verbose=1
)

# --------------------------------------------------
# Save model
# --------------------------------------------------
model.save("models/multimodal_drift_lstm.h5")

print("✅ Multimodal LSTM drift model saved at models/multimodal_drift_lstm.h5")
