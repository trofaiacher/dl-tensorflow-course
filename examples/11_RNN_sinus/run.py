import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
from pathlib import Path

# Add project root to Python path (so "shared" can be imported)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from matplotlib.lines import Line2D

from shared.seeds import set_seed
from shared.plotting import (
    savefig,
    COL_BLUE, COL_BLACK, COL_ORANGE, COL_RED,
    COL_GREEN, COL_GREY
)


# =============================
# 0) Reproducibility & Configuration
# =============================
SEED = 123
set_seed(SEED)

WINDOW = 40
HORIZON = 1
N = 1200
TRAIN_FRAC = 0.8
BATCH_SIZE = 32
EPOCHS = 15
NOISE_STD = 0.05   # 0.0 = perfect sine wave


# =============================
# 1) Generate time series
# =============================
t = np.linspace(0, 12*np.pi, N)
y = np.sin(t) + NOISE_STD * np.random.normal(size=N)

plt.figure(figsize=(10, 3.5))
plt.plot(t, y, linewidth=1.8, color=COL_BLUE)
plt.title("Time Series: Sine with Noise")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.grid(True, linestyle=":", linewidth=0.8, color=COL_GREY)
savefig("ts_sine_raw.png")
plt.close()


# =============================
# 2) Sliding window dataset
# =============================
def make_supervised(series: np.ndarray, window: int, horizon: int):
    X, Y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i:i+window])
        Y.append(series[i+window+horizon-1])
    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (samples, window, 1)
    Y = np.array(Y, dtype=np.float32)[..., np.newaxis]  # (samples, 1)
    return X, Y

X, Y = make_supervised(y, WINDOW, HORIZON)

split = int(len(X) * TRAIN_FRAC)  # chronological split
X_train, Y_train = X[:split], Y[:split]
X_val,   Y_val   = X[split:], Y[split:]

print("Shapes:")
print("X_train:", X_train.shape, "Y_train:", Y_train.shape)
print("X_val  :", X_val.shape,   "Y_val  :", Y_val.shape)

# Map time axis for validation targets
start_idx = WINDOW + HORIZON - 1 + split
t_val = t[start_idx : start_idx + len(Y_val)]
y_true = Y_val[:, 0]


# =============================
# 3) Model 1: LSTM
# =============================
def build_lstm(window: int):
    inp = layers.Input(shape=(window, 1), name="input")
    x = layers.LSTM(32, name="lstm")(inp)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    out = layers.Dense(1, name="out")(x)
    m = models.Model(inp, out, name="LSTM_forecaster")
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

lstm_model = build_lstm(WINDOW)
lstm_model.summary()

hist_lstm = lstm_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# =============================
# 4) Model 2: GRU
# =============================
def build_gru(window: int):
    inp = layers.Input(shape=(window, 1), name="input")
    x = layers.GRU(32, name="gru")(inp)
    x = layers.Dense(32, activation="relu", name="dense1")(x)
    out = layers.Dense(1, name="out")(x)
    m = models.Model(inp, out, name="GRU_forecaster")
    m.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return m

gru_model = build_gru(WINDOW)
gru_model.summary()

hist_gru = gru_model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# =============================
# 5) Plot: Training history (validation loss comparison)
# =============================
plt.figure(figsize=(7, 3.8))
plt.plot(hist_lstm.history["val_loss"], color=COL_GREEN, linewidth=2, label="LSTM val_loss")
plt.plot(hist_gru.history["val_loss"],  color=COL_ORANGE, linewidth=2, label="GRU val_loss")
plt.title("Comparison: Validation Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True, linestyle=":", linewidth=0.8, color=COL_GREY)
plt.legend()
savefig("ts_compare_val_loss.png")
plt.close()


# =============================
# 6) Predictions on validation range
# =============================
y_pred_lstm = lstm_model.predict(X_val, verbose=0)[:, 0]
y_pred_gru  = gru_model.predict(X_val, verbose=0)[:, 0]

mse_lstm = float(np.mean((y_true - y_pred_lstm)**2))
mse_gru  = float(np.mean((y_true - y_pred_gru )**2))
mae_lstm = float(np.mean(np.abs(y_true - y_pred_lstm)))
mae_gru  = float(np.mean(np.abs(y_true - y_pred_gru )))

print(f"[VAL] LSTM: MSE={mse_lstm:.6f} | MAE={mae_lstm:.6f}")
print(f"[VAL]  GRU: MSE={mse_gru :.6f} | MAE={mae_gru :.6f}")


# =============================
# 7) Plot: True vs LSTM vs GRU (overlay)
# =============================
plt.figure(figsize=(10, 3.9))
plt.plot(t_val, y_true,      color=COL_BLACK, linewidth=2.2, label="Ground truth (val)")
plt.plot(t_val, y_pred_lstm, color=COL_GREEN, linewidth=2.0, label=f"LSTM (MSE={mse_lstm:.4f})")
plt.plot(t_val, y_pred_gru,  color=COL_ORANGE, linewidth=2.0, label=f"GRU (MSE={mse_gru:.4f})")
plt.title("Validation Forecast: LSTM vs GRU")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, linestyle=":", linewidth=0.8, color=COL_GREY)
plt.legend()
savefig("ts_compare_val_pred_overlay.png")
plt.close()


# =============================
# 8) Plot: Zoom (last 200 points)
# =============================
ZOOM = 200 if len(t_val) > 200 else len(t_val)

plt.figure(figsize=(10, 3.9))
plt.plot(t_val[-ZOOM:], y_true[-ZOOM:],      color=COL_BLACK, linewidth=2.2, label="Ground truth (val)")
plt.plot(t_val[-ZOOM:], y_pred_lstm[-ZOOM:], color=COL_GREEN, linewidth=2.0, label="LSTM")
plt.plot(t_val[-ZOOM:], y_pred_gru[-ZOOM:],  color=COL_ORANGE, linewidth=2.0, label="GRU")
plt.title(f"Zoom: Last {ZOOM} Validation Points")
plt.xlabel("t")
plt.ylabel("y")
plt.grid(True, linestyle=":", linewidth=0.8, color=COL_GREY)
plt.legend()
savefig("ts_compare_val_pred_zoom.png")
plt.close()
