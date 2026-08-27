"""Environment Variables for the Visual Geoloc Package.
Most come from the .env file."""

import os



######## VARIABLES ##########
GEOCELL_SIZE = int(os.environ.get("GEOCELL_SIZE"))
LON_MIN = int(os.environ.get("LON_MIN"))
LON_MAX = int(os.environ.get("LON_MAX"))
LAT_MIN = int(os.environ.get("LAT_MIN"))
LAT_MAX = int(os.environ.get("LAT_MAX"))

CLASS_NUMBER =int(os.environ.get("CLASS_NUMBER"))

BUCKET_NAME = os.environ.get("BUCKET_NAME")
