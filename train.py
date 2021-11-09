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

def send_message_to_slack(url, acc, loss, training_time, model_path): 
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "학습이 완료되었습니다."
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Accuracy:*\n{acc}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Training Time:*\n{training_time}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Loss:*\n{loss}"
                    },{
                        "type": "mrkdwn",
                        "text": f"*gsutil URI:*\n{model_path}"
                    }
                ]
            }
        ]
    }
    requests.post(url, json=payload)

def request_deploy_api(model_path):
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    workflow_id = os.getenv("GITHUB_WORKFLOW")
    access_token = os.getenv("GITHUB_TOKEN")
    model_tag = os.getenv("MODEL_TAG")

    headers = {'Authorization' : 'token ' + access_token }
    data = {"ref": "main", "input":{"model_path": model_path, "model_tag": model_tag }}
    r = requests.post(f"http://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches", headers=headers, data=data)
    print(r)


def main():

    start = time.time()

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
    model.save(save_path)

    gs_path = os.path.join(model_path, save_path)

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