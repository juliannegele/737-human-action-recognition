import streamlit as st
import requests
from google.cloud import storage
import cv2
import tempfile
import pickle
import time

CSS = """
.st-al {
    color: rgb(23, 108, 54);
}
.stButton button {
    color: rgb(23, 108, 54);
    background-color: rgba(9, 171, 59, 0.2);
    border-color: rgba(9, 171, 59, 0.2);
    width: 100%;
}
.stButton button:hover {
    border-color: rgba(9, 171, 59, 0.2);
    color: #fff;
}
.stButton button:focus:not(:active) {
    border-color: white;
    color: #fff;
    background-color: white;

}
.stButton button:focus {
    box-shadow: none;
}

.st-al {
    background-color: rgba(9, 171, 59, 0.2);
}
h1 {
    background-color: rgb(240, 242, 246);
    margin-top: -120px;
    margin-left: -580px;
    margin-right: -580px;
    text-align: center;
}
video {
    width: 300px;
    height: 300px;
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
st.title("Human Action Recognition")


SEQUENCE_LENGTH = 10
IMAGE_HEIGHT = 64

IMAGE_WIDTH = 64

st.header("Let's predict some actions taking place in your personal video!")
st.subheader("Start by clicking the 'Browse files' button")
h = st.file_uploader(" ",type=[".mp4"])
if h is not None:
    st.markdown('''
        <style>
            .uploadedFile {display: none}
        <style>''',
                unsafe_allow_html=True)
    st.video(h)
    if st.button("Get Prediction"):


        with st.spinner('Your video is being analyzed'):
            time.sleep(5)
            #st.success('Done!')

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(h.read())


        #extracting the frames
        frames_list = []

        predicted_class_name = ""

        video_reader = cv2.VideoCapture(tfile.name)

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

        @st.cache(suppress_st_warning=True)
        def get_pred():
            pred = requests.get(url,params="frames.pkl").json()

            return pred
        prediction = get_pred()
        st.success(f"Done! - Your video is about: {prediction[18:]}")

        #if f"{prediction[19:]}" in h.name:
        #   st.balloons()
else:
    st.info("Don't let the kitten wait!")
    st.image("test_image.jpeg")
