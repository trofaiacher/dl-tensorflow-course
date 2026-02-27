import tensorflow as tf


def main():
    print("Example 01: MNIST MLP")
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()
