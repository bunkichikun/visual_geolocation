"""Environment Variables for the Visual Geoloc Package.
Most come from the .env file."""

import os



######## VARIABLES ##########
GEOCELL_SIZE = os.environ.get("GEOCELL_SIZE")
LON_MIN = os.environ.get("LON_MIN")
LON_MAX = os.environ.get("LON_MAX")
LAT_MIN = os.environ.get("LAT_MIN")
LAT_MAX = os.environ.get("LAT_MAX")

CLASS_NUMBER = os.environ.get("CLASS_NUMBER")
BATCH_SIZE = os.environ.get("BATCH_SIZE")
IMG_FOLDER = os.environ.get("IMG_FOLDER")
