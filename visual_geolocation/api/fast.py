import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

from visual_geolocation.ml_logic.registry import load_model
from visual_geolocation.utils import geocell_to_class, coord_to_geocell, class_to_geocell, geocell_to_coord, haversine, geoscore
from visual_geolocation.ml_logic.data import load_test_data_from_bucket
from visual_geolocation.ml_logic.model import predict_by_path



app = FastAPI()
app.state.model = load_model()
assert app.state.model is not None

app.state.test_df = load_test_data_from_bucket()
assert app.state.test_df is not None



# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


def get_coord_from_id(test_df, challenge_id):
    try:
        #print(test_df.head())
        #print(test_df.loc[0,"id"]==int(challenge_id))
        #print(f"challenge_id={challenge_id}")
        y = test_df[test_df["id"]==int(challenge_id)].set_index('id')
        #print(y)
        #print(f"y['longitude'] = {y['longitude']}, y['latitude'] = {y['latitude']}")
        return y.loc[int(challenge_id), "longitude"], y.loc[int(challenge_id), "latitude"]
    except:
        raise ValueError("❌ Unknown Challenge {id}")


# http://127.0.0.1:8000/evaluate?guessed_longitude=-73.950655&guessed_latitude=40.783282&challenge_id=2
@app.get("/evaluate")
def predict(
        guessed_longitude: float,    # -73.950655
        guessed_latitude: float,     # 40.783282
        challenge_id: str
    ):      # 1
    """
    Expects:
    * the human prediction in
         - longitude       e.g.    -73.950655
         - and latitude    e.g.    40.783282
    * and the challenge number (id of the picture)  e.g. 293174685725801

    Returns the human prediction to the server, which will return:
    * the human predcition (long & lat)
    * the machine prediction (long & lat)
    * the true location (long & lat)

    """
    y_human = geocell_to_class(coord_to_geocell(guessed_longitude, guessed_latitude))
    coord_from_test = get_coord_from_id(app.state.test_df, challenge_id)
    y_true = geocell_to_class(coord_to_geocell(coord_from_test[0], coord_from_test[1]))

    y_pred = predict_by_path(app.state.model, f"data_for_front/preprocessed/{challenge_id}_pp.png")  ## TODO make the prediction right
    #print(y_pred)
    #print(int(tf.math.argmax(y_pred, axis=1)))
    coord_machine = geocell_to_coord( class_to_geocell(int(tf.math.argmax(y_pred, axis=1))))

    human_haversine = haversine(guessed_longitude, guessed_latitude, coord_from_test[0], coord_from_test[1])
    human_geoscore = geoscore(human_haversine)
    machine_haversine = haversine(coord_machine[0], coord_machine[1],  coord_from_test[0], coord_from_test[1])
    machine_geoscore = geoscore(machine_haversine)

    human = {"human_lon":guessed_longitude, "human_lat":guessed_latitude, "human_haversine":human_haversine, \
        "human_geoscore":human_geoscore, "human_true_geocell":y_human==y_true}
    machine = {"machine_lon":coord_machine[0], "machine_lat":coord_machine[1], "machine_haversine":machine_haversine, \
        "machine_geoscore":machine_geoscore, "machine_true_geocell": int(tf.math.argmax(y_pred, axis=1))==y_true}
    true_coord = {"true_lon":coord_from_test[0] , "true_lat":coord_from_test[1]}


    return [human, machine, true_coord]




@app.get("/")
def root():
    return {  'ping': 'pong'}
