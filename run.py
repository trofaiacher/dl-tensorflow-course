## Run in Google Colab

Open the master notebook:
https://colab.research.google.com/github/trofaiacher/dl-tensorflow-course/blob/main/colab/MASTER_NOTEBOOK.ipynb

import os
import tensorflow as tf

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # quieter TF logs

    print("TensorFlow version:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPU available:", bool(gpus))
    if gpus:
        print("GPU(s):", gpus)

    # Tiny deterministic “heartbeat” computation
    x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    y = tf.matmul(x, x)
    print("Sanity matmul:\n", y.numpy())

    print("\nDeep Learning course pipeline ready.")

if __name__ == "__main__":
    main()
