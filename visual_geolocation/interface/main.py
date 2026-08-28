import numpy as np
import pandas as pd
from pathlib import Path
from google.cloud import storage
import tensorflow as tf

from visual_geolocation.ml_logic.registry import *

from visual_geolocation.ml_logic.preprocessing import build_labeled_dataframe, make_tf_dataset, preprocess_offline_one_folder
from visual_geolocation.ml_logic.model import initialize_model, compile_model, train_model
#from visual_geolocation.interface.workflow import
from visual_geolocation.utils import haversine, geoscore, coord_to_geocell, geocell_to_country
from visual_geolocation.ml_logic.data import get_data_with_cache , get_json, get_pickle
from visual_geolocation.params import IMG_FOLDER, CLASS_NUMBER, BATCH_SIZE, BUCKET_NAME, RAW_DATA_PATH, TRAIN_FILE, TEST_FILE, IMAGE_SIZE







def load_data_from_bucket():
    train_cache_path = Path(RAW_DATA_PATH).joinpath(TRAIN_FILE)
    test_cache_path = Path(RAW_DATA_PATH).joinpath(TEST_FILE)

    df_train = get_data_with_cache(
        bucket_name=BUCKET_NAME,
        source_blob_name=TRAIN_FILE,
        cache_path=train_cache_path,
    )

    df_test = get_data_with_cache(
        bucket_name=BUCKET_NAME,
        source_blob_name=TEST_FILE,
        cache_path=test_cache_path,
    )
    return df_train, df_test


def preprocess(train_df):
    df_subset = build_labeled_dataframe(train_df, IMG_FOLDER)
    return df_subset


def preprocess_offline():

    train_df, test_df = load_data_from_bucket()

    for i in range(0,9):
        img_folder = f"0{i}.zip"
        df_subset = build_labeled_dataframe(train_df, img_folder)

        preprocess_offline_one_folder(df_subset, img_folder)



def train():

    # train_df, test_df = load_data_from_bucket()

    # df_subset = preprocess(train_df)

    # dataset = make_tf_dataset(
    #     df_subset,
    #     IMG_FOLDER,
    #     img_size=(64, 64),
    #     batch_size=BATCH_SIZE
    # )

    dataset = tf.keras.utils.image_dataset_from_directory(
            "gs://visual-geolocation-osv5m/preprocessed/train/XP/",
            labels="inferred",
            image_size=(IMAGE_SIZE,IMAGE_SIZE),
            batch_size=BATCH_SIZE)

    model = initialize_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model = compile_model(model)
    model, history = train_model(model, dataset)

    return model, history



def evaluate ():
  pass


def predict (X_pred : pd.DataFrame = None) -> np.ndarray:
  pass


def evaluate_baseline():
    NB_EXPERIMENTS_BASELINE = 10_000

    test_df  = pd.read_csv("raw_data/test_lite.csv")

    results = {"random_geoscore":[],
               "random_haversine":[],
               "random_accuracy":[],
               "most_frequent_geoscore":[],
               "most_frequent_haversine":[],
               "most_frequent_accuracy":[]}

    for i in range(NB_EXPERIMENTS_BASELINE):
        res = evaluate_random(test_df)
        results["random_geoscore"].append(res[0])
        results["random_haversine"].append(res[1])
        results["random_accuracy"].append(res[2])


        res = evaluate_most_frequent(test_df)
        results["most_frequent_geoscore"].append(res[0])
        results["most_frequent_haversine"].append(res[1])
        results["most_frequent_accuracy"].append(res[2])

    print(f"""✅ Results for 🎲 Random Policy after {NB_EXPERIMENTS_BASELINE} Runs:
 * Mean Geoscore: {round(np.mean(results["random_geoscore"]), 2)}
 * Mean Haversine: {round(np.mean(results["random_haversine"]),2)}
 * Mean Country Accuracy {round(np.mean(results["random_accuracy"]),2)}

✅ Results for 🏆 "Most Frequent Class" Policy after {NB_EXPERIMENTS_BASELINE} Runs:
 * Mean Geoscore: {round(np.mean(results["most_frequent_geoscore"]),2)}
 * Mean Haversine: {round(np.mean(results["most_frequent_haversine"]),2)}
 * Mean Country Accuracy {round(np.mean(results["most_frequent_accuracy"]),2)}
""")



def evaluate_random(test_df):
    """Returns the Geoscore, Haversine distance and Country Accuracy of a random selected class"""

    t_i = np.random.randint(test_df.shape[0])
    target_lon, target_lat = test_df.loc[t_i , "longitude"], test_df.loc[t_i , "latitude"]
    p_i = np.random.randint(test_df.shape[0])
    pred_lon, pred_lat = test_df.loc[p_i , "longitude"], test_df.loc[p_i , "latitude"]

    d_haversine = haversine(target_lon, target_lat, pred_lon, pred_lat)

    s_geoscore = geoscore(d_haversine)

    # TODO Later, when test_final.csv is available
    # return 1 if the predicted country is right, 0 else

    #pred_country = geocell_to_country(coord_to_geocell(pred_lon, pred_lat))
    #target_country = test_df.loc[t_i, "unique_country"]

    #class_to_geocell(0)

    print(f'the traget is : {target_country}, and the prediction is : {pred_country}')

    #TODO
    # accuracy = target_country == pred_country but check that the country codes are the same...
    accuracy = np.nan

    return s_geoscore, d_haversine, accuracy



def evaluate_most_frequent(test_df):
    """Returns the Geoscore, Haversine distance and Country Accuracy of classification
    always predicting the most frequent class in training"""
    MOST_FREQUENT_CLASS = 42

    t_i = np.random.randint(test_df.shape[0])
    target_lon, target_lat = test_df.loc[t_i , "longitude"], test_df.loc[t_i , "latitude"]
    pred_lon, pred_lat = test_df.loc[MOST_FREQUENT_CLASS , "longitude"], test_df.loc[MOST_FREQUENT_CLASS , "latitude"]

    d_haversine = haversine(target_lon, target_lat, pred_lon, pred_lat)

    s_geoscore = geoscore(d_haversine)

    accuracy = np.nan # TODO Later, when test.csv is available

    return s_geoscore, d_haversine, accuracy



if __name__ == '__main__':

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob("train/00.zip")
    blob.download_to_filename("00.zip")

    model, history = train()
    print("test fonction main")
