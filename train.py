from typing import get_args
import tensorflow as tf
import argparse

class MnistClissifier(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(10, activation='softmax')
        

    def call(self, input):
        x = self.flatten(input)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

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

def main():
    args = get_args()
    epochs = args.epochs

    model = MnistClissifier()
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    
    # train_x, test_x = train_x/255.0, test_x/255.0
    
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(train_x, train_y, epochs=epochs)
    loss, acc = model.evaluate(test_x, test_y)
    print("model acc: {:.4f}, model loss: {:.4f}".format(acc, loss))

if __name__ == '__main__':
  main()