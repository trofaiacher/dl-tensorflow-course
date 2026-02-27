import os
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main():
    print("\n=== Example 01: MNIST MLP Baseline ===\n")

    set_seed(123)

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalise
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nTraining...\n")

    model.fit(
        x_train,
        y_train,
        epochs=3,
        batch_size=128,
        verbose=2
    )

    print("\nEvaluating...\n")

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    print(f"\nFinal test accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
