"""
All the works around the predicion model
"""
import datetime
import numpy as np
from keras import Model, Sequential, layers, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from typing import Tuple
from colorama import Fore, Style
from visual_geolocation.params import BATCH_SIZE, CLASS_NUMBER, EPOCHS, VAL_SPLIT
from visual_geolocation.utils import geocell_to_coord, class_to_geocell



def init_class_to_coord():
    """Construit deux tenseurs denses (lat, lon) indexés par numéro de classe.
    class_to_lat[i] / class_to_lon[i] donnent les coordonnées du centre de la classe i.

    Remplace l'ancienne implémentation basée sur tf.lookup.StaticHashTable :
    ces tables sont des ressources CPU-only, incompatibles avec la compilation
    XLA sur GPU (erreur "resource located in device CPU:0 from device GPU:0").
    tf.gather sur un tenseur dense fonctionne nativement sur GPU et avec XLA.
    """
    class_to_coord_map = [geocell_to_coord(class_to_geocell(i)) for i in range(CLASS_NUMBER)]

    class_to_lat = [c[1] for c in class_to_coord_map]
    class_to_lon = [c[0] for c in class_to_coord_map]

    lat_tensor = tf.constant(class_to_lat, dtype=tf.float32)
    lon_tensor = tf.constant(class_to_lon, dtype=tf.float32)

    return lat_tensor, lon_tensor


CLASS_TO_LAT_TENSOR, CLASS_TO_LON_TENSOR = init_class_to_coord()


@tf.keras.utils.register_keras_serializable()
def haversine_metric(y_true, y_pred):
    """Distance haversine moyenne (en km) entre la classe réelle et la classe prédite,
    en utilisant les coordonnées du centre de chaque géocell.
    """
    R = 6371.0

    y_true_idx = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred_idx = tf.cast(tf.math.argmax(y_pred, axis=-1), tf.int32)

    lat1 = tf.gather(CLASS_TO_LAT_TENSOR, y_true_idx)
    lon1 = tf.gather(CLASS_TO_LON_TENSOR, y_true_idx)
    lat2 = tf.gather(CLASS_TO_LAT_TENSOR, y_pred_idx)
    lon2 = tf.gather(CLASS_TO_LON_TENSOR, y_pred_idx)

    # Conversion degrés -> radians
    lat1 = lat1 * (3.141592653589793 / 180.0)
    lon1 = lon1 * (3.141592653589793 / 180.0)
    lat2 = lat2 * (3.141592653589793 / 180.0)
    lon2 = lon2 * (3.141592653589793 / 180.0)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Formule de haversine
    a = tf.math.sin(dlat / 2.0) ** 2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon / 2.0) ** 2

    # Stabilisation numérique pour éviter les NaN
    c = 2.0 * tf.math.asin(tf.math.sqrt(tf.clip_by_value(a, 0.0, 1.0)))

    return tf.reduce_mean(R * c)


def initialize_model(input_shape: tuple) -> Model:

    model = Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1./255))

    # Bloc 1
    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    
    # Bloc 2
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    
    # Bloc 3
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
    model.add(layers.GlobalAveragePooling2D())    
    model.add(layers.Dropout(0.2))

    # Flatten
    model.add(layers.Flatten())
    
    ### Fully Connected layers
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(0.3))

    ### Fully Connected layers-2
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.2))

    ### Fully Connected layers-3
    model.add(layers.Dense(512, activation='relu'))
     
    ### Last layer - Classification Layer with CLASS_NUMBER outputs
    model.add(layers.Dense(CLASS_NUMBER, activation='softmax'))

    return model


def compile_model(model : Model) -> Model:
    loss = losses.SparseCategoricalCrossentropy(ignore_class=-1)

    model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy', haversine_metric])

    return model


def train_model(
        model: Model,
        train_dataset,
        val_dataset,
        batch_size=BATCH_SIZE
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    checkpoints = ModelCheckpoint(
        f"checkpoint_model_{datetime.datetime.now().strftime('%m_%d_%H:%M')}.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stop = EarlyStopping(
        patience=25,
        restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoints, early_stop],
        batch_size=batch_size,
        verbose=1
    )

    print(f"✅ Model trained with max accuracy: {round(np.max(history.history['accuracy']), 2)}")

    return model, history


def predict_by_id(pic_id):
    predicted_class = 42
    return predicted_class
