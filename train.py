from typing import get_args
import tensorflow as tf
import argparse
import os
import datetime
from tensorflow.python.lib.io import file_io

def get_args():
    parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
  
    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        metavar='N',
        help='number of epochs to train (default: 1)')

    args = parser.parse_args()
    return args

def get_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax'),
        ]
    )
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    args = get_args()
    epochs = args.epochs

    gcp_bucket = "mnist_model_store"

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join("gs://", gcp_bucket, "mnist_model")
    # save_path = os.path.join("gs://", gcp_bucket, "mnist_model1", "save_at_{}.h5".format(timestamp))
    save_path = "save_at_{}.h5".format(timestamp)
    print(save_path)

    model = get_model()
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    
    # train_x, test_x = train_x/255.0, test_x/255.0
    
    history = model.fit(train_x, train_y, epochs=epochs)
    loss, acc = model.evaluate(test_x, test_y)
    print("model acc: {:.4f}, model loss: {:.4f}".format(acc, loss))

    # model.save(save_path)
    model.save(save_path)

    with file_io.FileIO(save_path, mode='rb') as input_file:
        with file_io.FileIO(os.path.join(model_path, save_path), mode='wb+') as output_file:
            output_file.write(input_file.read())

if __name__ == '__main__':
  main()