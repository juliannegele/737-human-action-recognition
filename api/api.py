import os

import cv2

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from google.cloud import storage
#from googleapiclient.discovery import build

from tensorflow.keras.models import load_model

from api.params import *

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


@app.get("/predict")
def predict(file_name):

    # downloading the video
    client = storage.Client.from_service_account_json("/Users/philkolling/code/philkolling/gcp/peppy-webbing-332911-743c3173bc03.json")
    bucket = client.get_bucket('737-human-action-recognition-bucket')

    blob = bucket.blob(f"{GCP_PATH}/{file_name}")

    blob.download_to_filename(f"api/video/{file_name}")

    video_reader = cv2.VideoCapture(f"api/video/{file_name}")

    #original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    #original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))


    SEQUENCE_LENGTH = 10
    #extracting the frames
    frames_list = []

    #predicted_class_name = ""

    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),10)

    for frame_counter in range(SEQUENCE_LENGTH):

        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter*skip_frames_window)

        success, frame = video_reader.read()

        if not success:
            break
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        normalized_frame = resized_frame /255

        frames_list.append(normalized_frame)

    #return print(frames_list)

    # predicting
    model = load_model("model_test.h5")

    predicted_labels_probabilites = model.predict(np.expand_dims(frames_list, axis=0))[0]

    predicted_label = np.argmax(predicted_labels_probabilites)

    predicted_class_name = CLASSES_LIST[predicted_label]

    video_reader.release()

    os.remove(f"api/video/{file_name}")

    return (f"predicted_class : {predicted_class_name}")
