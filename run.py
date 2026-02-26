import os
import json
import yaml
import random
import numpy as np
import tensorflow as tf


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_seed(seed)

    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print("Config:", config)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(config["model"]["hidden_units"][0], activation="relu"),
        tf.keras.layers.Dense(config["model"]["hidden_units"][1], activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config["train"]["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=config["train"]["epochs"],
        batch_size=config["train"]["batch_size"],
        verbose=2
    )

    loss, acc = model.evaluate(x_test, y_test, verbose=0)

    os.makedirs(config["output_dir"], exist_ok=True)

    results = {
        "loss": float(loss),
        "accuracy": float(acc),
        "seed": seed
    }

    with open(os.path.join(config["output_dir"], "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Final test accuracy:", acc)


if __name__ == "__main__":
    main()
