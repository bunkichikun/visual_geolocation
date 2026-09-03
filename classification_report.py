import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import classification_report

from visual_geolocation.params import TRAIN_SET_PATH, IMAGE_SIZE, BATCH_SIZE, VAL_SPLIT
from visual_geolocation.ml_logic.model import haversine_metric, initialize_model
from visual_geolocation.utils import geocell_to_class

model = initialize_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
assert model is not None

model.load_weights("./registry/models/20260902-141622.h5")
#print("model loaded")
#print(model)

from pathlib import Path

dossier_racine = Path("./preprocessed/test/256/")
extensions = {".png"}

y_pred = []
y_true = []

for class_id in range(3):#1443):
    print(f"starting class {class_id}")
    dossier_racine = Path("./preprocessed/test/256/").joinpath(f"{class_id}")

    #print(f"dossier racine={dossier_racine}")
    liste_images= [
        str(fichier)
        for fichier in dossier_racine.iterdir()
        if fichier.suffix.lower() in extensions
    ]

    liste_images_to_predict = []
    for chemin in liste_images:
        image_path=chemin
        img = Image.open(image_path)
        img_array = tf.keras.utils.img_to_array(img)

        liste_images_to_predict.append(img_array)

    #print(liste_images_to_predict)

    X_processed = tf.stack(liste_images_to_predict)
    print(f"Let's predict for class {class_id}!!")
    y_pred_proba = model.predict(X_processed)
    #print("Prediction done!!")
    #y_pred=np.stack([y_pred, np.argmax(Y_pred_proba,axis=1)], axis=1)
    y_pred_one_class = np.argmax(y_pred_proba,axis=0)
    y_pred = np.append(y_pred,  y_pred_one_class)
    #print("Stack y_pred")
    #y_true =np.stack([y_true, class_id * np.ones(shape=y_pred.shape, dtype=int)], axis=1)
    #y_true.append(class_id * np.ones(shape=y_pred_one_class.shape, dtype=int))
    y_true = np.append(y_true,  class_id * np.ones(shape=y_pred_one_class.shape, dtype=int))
    #print("Stack y_true")


with open('y_true.pickle', 'wb') as fp:
    pickle.dump(y_true, fp)

with open('y_pred.pickle', 'wb') as fp2:
    pickle.dump(y_pred, fp2)


#print(f"y_pred={y_pred}\n\ny_true={y_true}")
class_rep=classification_report(y_true, y_pred)

with open('class_rep.pickle', 'wb') as fp3:
    pickle.dump(y_pred, fp3)


print(f"""\n\nCLASSIFICATION REPORT\n\n{class_rep}""")


print(f"accuracy = {np.mean(y_true==y_pred)}")
