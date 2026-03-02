import sys
from pathlib import Path

# Add project root to Python path
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from shared.seeds import set_seed
from shared.plotting import savefig
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.lines import Line2D
# -----------------------------
# Colour-blind safe palette (Okabe–Ito)
# -----------------------------
COL_BLACK  = "#000000"
COL_ORANGE = "#E69F00"
COL_SKY    = "#56B4E9"
COL_GREEN  = "#009E73"
COL_BLUE   = "#0072B2"
COL_RED    = "#D55E00"
COL_PURPLE = "#CC79A7"
COL_GREY   = "#7F7F7F"

def savefig(name, dpi=300):
    plt.savefig(name, dpi=dpi, bbox_inches="tight")

# -----------------------------
# Daten: UND
# -----------------------------
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
y = np.array([[0],[0],[0],[1]], dtype=np.float32)

# -----------------------------
# Modell: 1 Neuron (Sigmoid)
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=1.0),
    loss="binary_crossentropy"
)

# -----------------------------
# Fehlklassifikationen pro Epoche
# -----------------------------
def miscls_count():
    y_hat = (model.predict(X, verbose=0) >= 0.5).astype(int)
    return int(np.sum(y_hat != y.astype(int)))

epochs = 200
mis_history = []

for _ in range(epochs):
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)
    mis_history.append(miscls_count())

# Plot: Epoche vs Fehlklassifikation
plt.figure(figsize=(6,4))
plt.plot(mis_history, linewidth=2, color=COL_BLUE)
plt.xlabel("Epoche")
plt.ylabel("Fehlklassifikation")
plt.title("UND: Fehlklassifikationen pro Epoche")
plt.grid(True, linestyle=":", linewidth=0.8)
savefig("und_fehlklassifikation.png")
plt.show()

# -----------------------------
# Plot: Datenpunkte + Trennlinie
# -----------------------------
xx, yy = np.meshgrid(np.linspace(-0.25, 1.25, 250),
                     np.linspace(-0.25, 1.25, 250))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
P = model.predict(grid, verbose=0).reshape(xx.shape)

plt.figure(figsize=(6,6))

# Datenpunkte in Okabe–Ito
mask0 = (y[:,0] == 0)
mask1 = (y[:,0] == 1)

plt.scatter(X[mask0,0], X[mask0,1],
            s=110, marker="o", color=COL_BLUE, edgecolors=COL_BLACK,
            linewidths=0.8, label="UND = 0")
plt.scatter(X[mask1,0], X[mask1,1],
            s=110, marker="s", color=COL_ORANGE, edgecolors=COL_BLACK,
            linewidths=0.8, label="UND = 1")

# Trennlinie (Kontur bei 0.5)
plt.contour(xx, yy, P, levels=[0.5], colors=[COL_RED], linewidths=2.5)

# Proxy-Handle für Legende der Trennlinie
line_proxy = Line2D([0], [0], color=COL_RED, lw=2.5, label="Trennlinie")
handles, labels = plt.gca().get_legend_handles_labels()
handles.append(line_proxy)
labels.append("Trennlinie")
plt.legend(handles, labels, loc="upper right")

plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("UND: Datenpunkte + Trennlinie")
plt.grid(True, linestyle=":", linewidth=0.8)

savefig("und_trennlinie.png")
plt.show()
