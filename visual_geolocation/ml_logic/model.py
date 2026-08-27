"""
All the works around the predicion model
"""
import numpy as np
from keras import Model, Sequential, layers
from typing import Tuple
from colorama import Fore, Style
from visual_geolocation.params import BATCH_SIZE, CLASS_NUMBER




def initialize_model(input_shape: tuple) -> Model:

    model = Sequential()

    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(8, (4, 4), activation="relu",padding='same'))

    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with NUMBER_OF_GOOD_GEOCEL outputs corresponding to NUMBER_OF_GOOD_GEOCEL digits
    model.add(layers.Dense(CLASS_NUMBER,activation='softmax'))

    return model


def compile_model(model : Model) -> Model:
    loss = keras.losses.SparseCategoricalCrossentropy(ignore_class=-1)

    model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy'])

    return model


def train_model(
        model: Model,
        dataset,
        batch_size=BATCH_SIZE
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    history = model.fit(
        dataset,
        epochs=1,
        batch_size=batch_size,
        verbose=1
    )

    print(f"✅ Model trained with max accuracy: {round(np.max(history.history['accuracy']), 2)}")

    return model, history
