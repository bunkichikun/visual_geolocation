"""Handy functions for the Visual Geolocation Package
"""
import pickle
import os

import geopandas as gpd
import numpy as np

from pathlib import Path

from shapely.geometry import Point

from visual_geolocation.params import GEOCELL_SIZE, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, RAW_DATA_PATH,\
    CLASS_TO_GEOCELL_MAP, IMAGES_PATH, BOUNDARIES_JSON, BUCKET_NAME
from visual_geolocation.ml_logic.data import get_pickle , get_json, load_data_from_bucket



EARTH_RADIUS = 6371
GEOSCORE_MODULE = 5000
GEOSCORE_FACTOR = 1492.7



load_data_from_bucket(BUCKET_NAME, RAW_DATA_PATH, CLASS_TO_GEOCELL_MAP, BOUNDARIES_JSON)


with open(os.path.join(RAW_DATA_PATH, CLASS_TO_GEOCELL_MAP), 'rb') as handle:
    CLASS_TO_GEOCELL = pickle.load(handle)


def coord_to_geocell(lon, lat, geocell_size = GEOCELL_SIZE):
    """Returns the index of the Geocell that contains the point of coordinates
    (lon, lat). A third optional argument is the geocell_size, which defaults
    to the environment variable GEOCELL_SIZE.

    Assumes the considered point is within the boundaries in the environment
    variables with LAT_MIN, LAT_MAX, LON_MIN and LON_MAX.
    """

    assert LON_MIN <= lon <= LON_MAX
    assert LAT_MIN <= lat <= LAT_MAX

    lon_bin = (lon - LON_MIN ) // geocell_size
    lat_bin = (lat - LAT_MIN ) // geocell_size

    return lon_bin * ((LAT_MAX - LAT_MIN) // geocell_size +1) + lat_bin


def geocell_to_coord(geocell_idx, geocell_size = GEOCELL_SIZE):
    """Given a Geocell index (the class identifier in our classification)
    returns the latitude and longitude of the Geocell centroid.
    """

    lat_bin = geocell_idx %  ((LAT_MAX - LAT_MIN) // geocell_size +1)
    lon_bin = geocell_idx //  ((LAT_MAX - LAT_MIN) // geocell_size +1)
    lon = LON_MIN + lon_bin * geocell_size + 0.5
    lat = LAT_MIN + lat_bin * geocell_size + 0.5

    return lon, lat


def geocell_to_country(geocell_idx):
    """Given a Geocell index (the class identifier in our classification)
    returns a tuple with the (country name, 3-letter-code) for the country
    where centroid of the geocell lies.

    return example ('France', 'FR1')
    """
    lon, lat = geocell_to_coord(geocell_idx)

    # geojson with countries from https://geojson-maps.kyd.au/
    countries_df = gpd.read_file(os.path.join(IMAGES_PATH, BOUNDARIES_JSON))

    # Create GeoDataFrame for the point
    point_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    )

    # Ensure CRS consistency
    countries_df = countries_df.to_crs(point_gdf.crs)

    # Spatial join
    result = gpd.sjoin(point_gdf, countries_df, predicate="within")

    if result.empty:
        return None, None  # ou une valeur par défaut style ("Ocean", "N/A")

    return result["sovereignt"].iloc[0], result["sov_a3"].iloc[0]


def class_to_geocell(class_idx):
    """Returns the geocell index corresponding to the class index. This is
    loaded from a pickle file"""
    assert 0<= class_idx <= len(CLASS_TO_GEOCELL)
    return CLASS_TO_GEOCELL[class_idx]


def geocell_to_class(geocell_idx):
    """Returns the Class index corresponding to the Geocell index. This is loaded from a pickle file"""
    if geocell_idx not in CLASS_TO_GEOCELL:
        return -1
    return CLASS_TO_GEOCELL.index(geocell_idx)



def haversine(lon_1, lat_1, lon_2, lat_2):
    """Computes the distance between two points (described by their latitude
    and longitude) at the surface of the earth"""
    lat_1_rad, lon_1_rad = np.radians(lat_1), np.radians(lon_1)
    lat_2_rad, lon_2_rad = np.radians(lat_2), np.radians(lon_2)

    dlon_rad = lon_2_rad - lon_1_rad
    dlat_rad = lat_2_rad - lat_1_rad

    a = np.sin(dlat_rad / 2.0)**2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * \
        np.sin(dlon_rad / 2.0)**2
    haversine_rad = 2 * np.arcsin(np.sqrt(a))

    return haversine_rad * EARTH_RADIUS


def geoscore(distance):
    """Returns the geoscore, as reverse-engineered in the litterature (cf. OSV5M)
    geoscore = 5000 * exp(- haversine_distance / 1492.7)
    """
    return GEOSCORE_MODULE * np.exp(-distance / GEOSCORE_FACTOR)
