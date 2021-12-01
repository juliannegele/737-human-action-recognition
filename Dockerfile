FROM python: 3.8.12-buster

COPY api / api
COPY 737-human-action-recognition / 737-human-action-recognition
COPY model_test.h5 / model_test.h5
COPY requirements.txt / requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn api.api:app --host 0.0.0.0 --port $PORT