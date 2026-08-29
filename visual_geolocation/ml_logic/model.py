"""
All the works around the predicion model
"""
import datetime
import numpy as np
from keras import Model, Sequential, layers, losses
from keras.callbacks import ModelCheckpoint, EarlyStopping
from typing import Tuple
from colorama import Fore, Style
from visual_geolocation.params import BATCH_SIZE, CLASS_NUMBER, EPOCHS, VAL_SPLIT




def initialize_model(input_shape: tuple) -> Model:

    model = Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Rescaling(1./255))

    # Bloc 1
    model.add(layers.Conv2D(16, (3, 3), activation="relu", padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # Bloc 2
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    # Bloc 3
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    ### Fully Connected layers
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))

    ### Last layer - Classification Layer with CLASS_NUMBER outputs
    model.add(layers.Dense(CLASS_NUMBER, activation='softmax'))

    return model


def compile_model(model : Model) -> Model:
    loss = losses.SparseCategoricalCrossentropy(ignore_class=-1)

    model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy'])

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
