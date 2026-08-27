import pandas as pd
from visual_geolocation.params import GCP_PROJECT, BUCKET_NAME
from pathlib import Path
from google.cloud import storage


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
