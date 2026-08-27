"""Environment Variables for the Visual Geoloc Package.
Most come from the .env file."""

import os

######## VARIABLES ##########
GEOCELL_SIZE = int(os.environ.get("GEOCELL_SIZE"))
LON_MIN = int(os.environ.get("LON_MIN"))
LON_MAX = int(os.environ.get("LON_MAX"))
LAT_MIN = int(os.environ.get("LAT_MIN"))
LAT_MAX = int(os.environ.get("LAT_MAX"))

MODEL_TARGET = os.environ.get("MODEL_TARGET")



## PATHS
GCP_PROJECT = os.environ.get("GCP_PROJECT")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH")
CLASS_TO_GEOCELL_MAP = os.environ.get("CLASS_TO_GEOCELL_MAP")
TRAIN_FILE = os.environ.get("TRAIN_FILE")
TEST_FILE = os.environ.get("TEST_FILE")
IMAGES_PATH = os.environ.get("IMAGES_PATH")
BOUNDARIES_JSON = os.environ.get("BOUNDARIES_JSON")
IMG_FOLDER = os.environ.get("IMG_FOLDER")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")


## TRAINING
CLASS_NUMBER = int(os.environ.get("CLASS_NUMBER"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE"))
