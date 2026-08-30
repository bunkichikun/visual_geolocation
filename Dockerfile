FROM python:3.10.6-slim

RUN pip install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip install  --no-cache-dir -r  requirements.txt

COPY visual_geolocation visual_geolocation

CMD uvicorn visual_geolocation.api.fast:app --reload --host 0.0.0.0 --port $PORT
