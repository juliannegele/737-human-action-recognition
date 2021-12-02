#import os

#import cv2

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage
#from googleapiclient.discovery import build

from tensorflow.keras.models import load_model

from api.params import *

import pickle

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def index():
    return dict(greeting = "Welcome!")


@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}


@app.get("/predict")
def predict(file_name):

    # downloading the frames_list
    client = storage.Client.from_service_account_json("/Users/philkolling/code/philkolling/gcp/peppy-webbing-332911-743c3173bc03.json")
    bucket = client.get_bucket('737-human-action-recognition-bucket')

    blob = bucket.blob(f"{GCP_PATH}/{file_name}")

    blob.download_to_filename(f"api/video/frames.pkl")


    # predicting
    model = load_model("model_test.h5")

    with open('frames.pkl', 'rb') as f:
        frames_list = pickle.load(f)

    predicted_labels_probabilites = model.predict(np.expand_dims(frames_list, axis=0))[0]

    predicted_label = np.argmax(predicted_labels_probabilites)

    predicted_class_name = CLASSES_LIST[predicted_label]

    #video_reader.release()

    #os.remove(f"api/video/{file_name}")

    return (f"predicted_class : {predicted_class_name}")
