import glob
import os
import time
import pickle

from visual_geolocation.params import LOCAL_REGISTRY_PATH, BUCKET_NAME, MODEL_TARGET
from colorama import Fore, Style
from tensorflow import keras
from google.cloud import storage
from visual_geolocation.ml_logic.model import haversine_metric, compile_model
from pathlib import Path


def save_results(params: dict, metrics: dict) -> None:


    timestamp = time.strftime("%Y%m%d-%H%M%S")

    params_path = Path(LOCAL_REGISTRY_PATH) / "params" / f"{timestamp}.pickle"
    metrics_path = Path(LOCAL_REGISTRY_PATH) / "metrics" / f"{timestamp}.pickle"

    # Créer les dossiers s'ils n'existent pas
    params_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Save params locally
    if params is not None:
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # Save metrics locally
    if metrics is not None:
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")


def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/{timestamp}.h5"
    - if MODEL_TARGET='gcs', also persist it in your bucket on GCS at "models/{timestamp}.h5" --> unit 02 only
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Créer les dossiers s'ils n'existent pas
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH,"models" ,f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!

        model_filename = model_path.split("/")[-1] # e.g. "20230208-161047.h5" for instance
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"models/{model_filename}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None


    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only

    Return None (but do not Raise) if no model is found

    """


    if MODEL_TARGET == "local":
        print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

        # Get the latest model version name by the timestamp on disk
        local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
        local_model_paths = glob.glob(f"{local_model_directory}/*")

        if not local_model_paths:
            return None

        most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

        print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

        latest_model = keras.models.load_model(most_recent_model_path_on_disk)

        print("✅ Model loaded from local disk")

        return latest_model

    elif MODEL_TARGET == "gcs":
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix="models"))

        try:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(LOCAL_REGISTRY_PATH, latest_blob.name)
            Path(latest_model_path_to_save).parent.mkdir(parents=True, exist_ok=True)
            latest_blob.download_to_filename(latest_model_path_to_save)

            latest_model = keras.models.load_model(latest_model_path_to_save)

            print("✅ Latest model downloaded from cloud storage")

            return latest_model
        except Exception as e:
            print(e)
            print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

            return None

    else:
        print(f"\n❌ Nothing to do in Model Load...")
        return None
