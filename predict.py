import os
import joblib
import pandas as pd
from google.cloud import storage
import gcsfs


PATH_TO_LOCAL_MODEL = 'model.joblib'
BUCKET_NAME = '737-human-action-recognition-bucket'
from keras.models import load_model
import h5py
import gcsfs
PROJECT_NAME = '737-human-action-recognition'
#CREDENTIALS = 'cred.json'
MODEL_PATH = '737-human-action-recognition-bucket/models/model_test.h5'
def download_video_test():
    FS = gcsfs.GCSFileSystem(project=PROJECT_NAME)
    with FS.open(MODEL_PATH, 'rb') as model_file:
        model_gcs = h5py.File(model_file, 'r')
        model = load_model(model_gcs)
    return model

def download_model(bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = 'models/model_test'
    blob = client.blob(storage_location)
    blob.download_to_filename('prediction_model')
    print("=> model downloaded from storage")
    model = joblib.load('prediction_model.joblib')
    if rm:
        os.remove('model.joblib')
    return model

def download_model_test():
    # Initialise a client
    storage_client = storage.Client("[737-human-action-recognition]")
    # Create a bucket object for our bucket
    bucket = storage_client.get_bucket(BUCKET_NAME)
    # Create a blob object from the filepath
    blob = bucket.blob("models/model")
    # Download the file to a destination
    blob.download_to_filename('test')


def load_joblib(BUCKET_NAME, file_name):
    fs = gcsfs.GCSFileSystem()
    with fs.open(f'{BUCKET_NAME}/models/{file_name}') as f:
        return joblib.load(f)
    print('model loaded')


def get_model(path_to_joblib):
    model = joblib.load(path_to_joblib)
    return model







if __name__ == '__main__':

    model = load_joblib(BUCKET_NAME,)
