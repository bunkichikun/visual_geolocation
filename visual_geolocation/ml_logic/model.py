"""
All the works around the predicion model
"""
import datetime
import numpy as np
from keras import Model, Sequential, layers, losses, backend
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from typing import Tuple
from colorama import Fore, Style
from visual_geolocation.params import BATCH_SIZE, CLASS_NUMBER, EPOCHS, VAL_SPLIT, IMAGE_SIZE
from visual_geolocation.utils import geocell_to_coord, class_to_geocell
from PIL import Image



def init_class_to_coord():
    class_to_coord_map = [ geocell_to_coord(class_to_geocell(i)) for i in range(CLASS_NUMBER)]

    keys_lat = tf.constant(range(CLASS_NUMBER), dtype=tf.int64)
    values_lat = tf.constant([c[1] for c in class_to_coord_map], dtype=tf.float32)
    table_lat = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_lat, values_lat),
        default_value=0.0)

    keys_lon = tf.constant(range(CLASS_NUMBER), dtype=tf.int64)
    values_lon = tf.constant([c[0] for c in class_to_coord_map], dtype=tf.float32)
    table_lon = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys_lon, values_lon),
        default_value=0.0)

    return table_lat, table_lon

## To reimplement the class_to_coord function
STATIC_CLASS_TO_LAT, STATIC_CLASS_TO_LON = init_class_to_coord()

@tf.keras.utils.register_keras_serializable()
def haversine_metric(y_true, y_pred):
    #tf.print(f"y_true.shape={y_true.shape}\ny_true={y_true}\ny_pred.shape={y_pred.shape}\ny_pred={y_pred}")
    #tf.print(y_true)
    R = 6371.0

    # Conversion from degrees to radians
    # y_true[:, 0] = latitude, y_true[:, 1] = longitude

    lat1 = STATIC_CLASS_TO_LAT.lookup(tf.cast(y_true, tf.int64))
    lon1 = STATIC_CLASS_TO_LON.lookup(tf.cast(y_true, tf.int64))
    lat2 = STATIC_CLASS_TO_LAT.lookup(tf.math.argmax(y_pred))
    lon2 = STATIC_CLASS_TO_LON.lookup(tf.math.argmax(y_pred))

    lat1 = lat1 * (3.141592653589793 / 180.0)
    lon1 = lon1 * (3.141592653589793 / 180.0)
    lat2 = lat2 * (3.141592653589793 / 180.0)
    lon2 = lon2 * (3.141592653589793 / 180.0)

    # Delta lat & long
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # haversine formula
    a = tf.math.sin(dlat / 2.0)**2 + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.sin(dlon / 2.0)**2

    # Numerical Stabilisation to avoid a NaN
    c = 2.0 * tf.math.asin(backend.sqrt(backend.clip(a, 0.0, 1.0)))
    # Returns mean distance over all batch
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


def predict_by_path(model, pic_path):

    images = []
    img = Image.open(pic_path)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    images.append(img_array)

    X_processed = tf.stack(images)
    y_pred = model.predict(X_processed)
    return y_pred
