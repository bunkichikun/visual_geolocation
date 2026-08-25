"""Handy functions for the Visual Geolocation Package
"""

from params import GEOCELL_SIZE, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX

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
    returns a tuple with the (country name and 3-letter-code) for the country where centroid of the
    geocell lies.
    """
    lon, lat = geocell_to_coord(geocell_idx)

    country = "toto"

    return  country
