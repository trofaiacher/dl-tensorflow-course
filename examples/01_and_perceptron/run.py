import os
# Force CPU to avoid GPU/PTX issues in Colab
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")  # render plots to files (no GUI)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def savefig(name: str, dpi: int = 300) -> None:
    plt.savefig(name, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {name}")


# -----------------------------
# Colour-blind safe palette (Okabe–Ito)
# -----------------------------
COL_BLACK  = "#000000"
COL_ORANGE = "#E69F00"
COL_BLUE   = "#0072B2"
COL_RED    = "#D55E00"


# -----------------------------
# Data: AND
# -----------------------------
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]], dtype=np.float32)

y = np.array([[0],
              [0],
              [0],
              [1]], dtype=np.float32)


# -----------------------------
# Model: single neuron (sigmoid)
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
# Misclassifications per epoch
# -----------------------------
def miscls_count() -> int:
    y_hat = (model.predict(X, verbose=0) >= 0.5).astype(int)
    return int(np.sum(y_hat != y.astype(int)))


# -----------------------------
# Train
# -----------------------------
epochs = 200
mis_history = []

for _ in range(epochs):
    model.fit(X, y, epochs=1, batch_size=4, verbose=0)
    mis_history.append(miscls_count())


# -----------------------------
# Plot: misclassifications
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(mis_history, linewidth=2, color=COL_BLUE)
plt.xlabel("Epoch")
plt.ylabel("Misclassifications")
plt.title("AND: Misclassifications per Epoch")
plt.grid(True, linestyle=":", linewidth=0.8)
savefig("and_misclassifications.png")
plt.close()


# -----------------------------
# Plot: decision boundary
# -----------------------------
xx, yy = np.meshgrid(np.linspace(-0.25, 1.25, 250),
                     np.linspace(-0.25, 1.25, 250))
grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
P = model.predict(grid, verbose=0).reshape(xx.shape)

plt.figure(figsize=(6, 6))

mask0 = (y[:, 0] == 0)
mask1 = (y[:, 0] == 1)

plt.scatter(X[mask0, 0], X[mask0, 1],
            s=110, marker="o", color=COL_BLUE,
            edgecolors=COL_BLACK, linewidths=0.8,
            label="AND = 0")

plt.scatter(X[mask1, 0], X[mask1, 1],
            s=110, marker="s", color=COL_ORANGE,
            edgecolors=COL_BLACK, linewidths=0.8,
            label="AND = 1")

plt.contour(xx, yy, P, levels=[0.5],
            colors=[COL_RED], linewidths=2.5)

line_proxy = Line2D([0], [0], color=COL_RED,
                    lw=2.5, label="Decision boundary")

handles, labels = plt.gca().get_legend_handles_labels()
handles.append(line_proxy)
labels.append("Decision boundary")

plt.legend(handles, labels, loc="upper right")
plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 1.25)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("AND: Data Points + Decision Boundary")
plt.grid(True, linestyle=":", linewidth=0.8)

savefig("and_decision_boundary.png")
plt.close()

print("Done.")
