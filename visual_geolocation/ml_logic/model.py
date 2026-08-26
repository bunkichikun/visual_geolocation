import numpy as np
from tensorflow import keras
from keras import Model, Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from typing import Tuple

"""
All the works around the predicion model
"""



def initialize_model(input_shape: tuple) -> Model:

    model.add(Input(shape=SIZE_OF_THE_PICTURE))
    model.add(layers.Conv2D(8, (4, 4), activation="relu",padding='same'))

    model.add(layers.MaxPool2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    ### One Fully Connected layer - "Fully Connected" is equivalent to saying "Dense"
    model.add(layers.Dense(10, activation='relu'))

    ### Last layer - Classification Layer with NUMBER_OF_GOOD_GEOCEL outputs corresponding to NUMBER_OF_GOOD_GEOCEL digits
    model.add(layers.Dense(NUMBER_OF_GOOD_GEOCELL,activation='softmax'))


    return model


def compile_model(model : Model) -> Model:
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    return model


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=BATCH_SIZE,
        patience=2,
        validation_data=None, # overrides validation_split
        validation_split=0.3
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        #monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=100,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history
