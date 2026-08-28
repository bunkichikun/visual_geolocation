import pandas as pd
from visual_geolocation.params import GCP_PROJECT, BUCKET_NAME, IMG_FOLDER, IMAGE_SIZE
from pathlib import Path
from google.cloud import storage
from colorama import Fore, Style
from keras.utils.image_utils import  array_to_img
import tensorflow as tf



def get_data_with_cache(bucket_name, source_blob_name, cache_path):
    """
    Retrieve data from local `cache_path` if the file already exists,
    otherwise download it from GCS bucket and store it at `cache_path`
    for future use.
    """

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad data from local CSV..." + Style.RESET_ALL)
        df = pd.read_csv(cache_path)
    else:
        print(Fore.BLUE + "\nLoad data from GCS bucket..." + Style.RESET_ALL)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(cache_path)

        df = pd.read_csv(cache_path)

    print(f"✅ Data loaded, with shape {df.shape}")

    return df


def get_json(bucket_name, source_json_name, cache_path):

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad JSON from local file..." + Style.RESET_ALL)
    else:
        print(Fore.BLUE + "\nLoad JSON from GCS bucket..." + Style.RESET_ALL)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_json_name)
        blob.download_to_filename(cache_path)

    print(f"✅ JSON ready at {cache_path}")



def get_pickle(bucket_name, source_json_name, cache_path):

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad pickle from local file..." + Style.RESET_ALL)
    else:
        print(Fore.BLUE + "\nLoad pickle from GCS bucket..." + Style.RESET_ALL)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(source_json_name)
        blob.download_to_filename(cache_path)

    print(f"✅ pickle ready at {cache_path}")


def get_zip_file(bucket_name, source_json_name, cache_path):

    if cache_path.is_file():
        print(Fore.BLUE + "\nLoad zip file from local file..." + Style.RESET_ALL)
    else:
        print(Fore.BLUE + "\nLoad zip file from GCS bucket..." + Style.RESET_ALL)

        cache_path.parent.mkdir(parents=True, exist_ok=True)

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(str(source_json_name))
        blob.download_to_filename(cache_path)

    print(f"✅ zip file ready at {cache_path}")


def load_data_from_bucket(BUCKET_NAME, RAW_DATA_PATH, CLASS_TO_GEOCELL_MAP, BOUNDARIES_JSON):

    pickle_path = Path(RAW_DATA_PATH).joinpath(CLASS_TO_GEOCELL_MAP)
    boundaries_json_path = Path(RAW_DATA_PATH).joinpath(BOUNDARIES_JSON)

    get_json(
        bucket_name=BUCKET_NAME,
        source_json_name=f"{RAW_DATA_PATH}/{BOUNDARIES_JSON}",
        cache_path=boundaries_json_path
    )

    get_pickle(
        bucket_name=BUCKET_NAME,
        source_json_name=f"{RAW_DATA_PATH}/{CLASS_TO_GEOCELL_MAP}",
        cache_path=pickle_path
    )


def dump_preprocessed_image(id, img_array, label, which):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    #print(img_array)
    encoded = tf.io.encode_png(img_array)
    blob = bucket.blob(f"preprocessed/{which}/{IMAGE_SIZE}/{str(label).split('.')[0]}/{id}_pp.png")
    blob.upload_from_string(encoded.numpy(), content_type="image/png")
    #print(f"dumped into preprocessed/train/{IMG_FOLDER.split('.')[0]}/{str(label).split('.')[0]}/{id}_pp{IMAGE_SIZE}.png !!!")
