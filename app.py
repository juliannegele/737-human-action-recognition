import streamlit as st
import requests
from google.cloud import storage
import cv2
import tempfile
import pickle
'''

# Human Action Recognition

'''

h = st.file_uploader("Upload file")
if h is not None:
    st.video(h)
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(h.read())



SEQUENCE_LENGTH = 10
IMAGE_HEIGHT = 64

IMAGE_WIDTH = 64
#extracting the frames
frames_list = []

predicted_class_name = ""

video_reader = cv2.VideoCapture(tfile.name)

video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
st.text(video_frames_count)
skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH),10)

for frame_counter in range(SEQUENCE_LENGTH):

    video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter*skip_frames_window)

    success, frame = video_reader.read()

    if not success:
        break
    resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

    normalized_frame = resized_frame /255

    frames_list.append(normalized_frame)

with open('frames.pkl', 'wb') as f:
    pickle.dump(frames_list, f)
#print(frames_list)

client = storage.Client.from_service_account_json(
    "/Users/philkolling/code/philkolling/gcp/peppy-webbing-332911-743c3173bc03.json")
bucket = client.get_bucket('737-human-action-recognition-bucket')

blob = bucket.blob("frontend_video/frames.pkl")
blob.upload_from_filename("frames.pkl")


#preprocessing in the app.py

url = "http://127.0.0.1:8000/predict?file_name="


pred = requests.get(url,params="frames.pkl").json()

st.text(f"{pred}")
