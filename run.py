import tensorflow as tf

def main():
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    print("Deep Learning course pipeline ready.")

if __name__ == "__main__":
    main()
