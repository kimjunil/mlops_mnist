from typing import get_args
import tensorflow as tf
import argparse
import os
import datetime
import requests
from tensorflow.python.lib.io import file_io
import time
import datetime
import requests
from utils import send_message_to_slack
from utils import request_deploy_api

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

    start = time.time()

    args = get_args()
    epochs = args.epochs
    gcp_bucket = os.getenv("GCS_BUCKET")

    bucket_path = os.path.join("gs://", gcp_bucket, "mnist_model")

    model = get_model()
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    
    # train_x, test_x = train_x/255.0, test_x/255.0
    
    history = model.fit(train_x, train_y, epochs=epochs)
    loss, acc = model.evaluate(test_x, test_y)
    print("model acc: {:.4f}, model loss: {:.4f}".format(acc, loss))

    timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    save_path = "save_at_{}_acc_{}_loss_.h5".format(timestamp, acc, loss)
    model.save(save_path)

    gs_path = os.path.join(bucket_path, save_path)

    with file_io.FileIO(save_path, mode='rb') as input_file:
        with file_io.FileIO(gs_path, mode='wb+') as output_file:
            output_file.write(input_file.read())

    end = time.time()
    sec = (end - start) 
    training_time = str(datetime.timedelta(seconds=sec)).split(".")[0]

    slack_url = os.getenv("WEB_HOOK_URL")
    if slack_url != None:
        send_message_to_slack(slack_url, acc, loss, training_time, gs_path)

    request_deploy_api(gs_path)
    
if __name__ == '__main__':
  main()