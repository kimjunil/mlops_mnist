from typing import get_args
import tensorflow as tf
import argparse
import os
import datetime
import requests

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

def send_message_to_slack(url, text): 
    payload = { "text" : text } 
    requests.post(url, json=payload)

def main():
    args = get_args()
    epochs = args.epochs

    gcp_bucket = "mnist_model_store"

    model_path = os.path.join("gs://", gcp_bucket, "mnist_model")

    model = get_model()
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    
    # train_x, test_x = train_x/255.0, test_x/255.0
    
    history = model.fit(train_x, train_y, epochs=epochs)
    loss, acc = model.evaluate(test_x, test_y)
    print("model acc: {:.4f}, model loss: {:.4f}".format(acc, loss))

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    save_path = "save_at_{}_acc_{}_loss_.h5".format(timestamp, acc, loss)
    model_path = os.path.join(model_path, save_path)
    model.save(model_path)

    slack_url = os.getenv("WEB_HOOK_URL")
    if slack_url != None:
        send_message_to_slack(slack_url, f"학습완료! , {model_path}")

if __name__ == '__main__':
  main()