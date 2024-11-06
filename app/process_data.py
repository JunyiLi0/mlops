# Importing required modules
import pandas as pd
from joblib import load
import os
import skimage
import shutil
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing import image as image_utils
import numpy as np
from PIL import Image

folder_path = "data/"
model = load('../best_model.joblib')
order = load('../label_order.joblib')

def read_image(file_name):
        image = skimage.io.imread(file_name)
        image = skimage.transform.resize(image, (64, 64, 1), mode='reflect')
        return image[:,:,:]


def predict_and_save():
    file_path = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Ajouter d'autres extensions si nécessaire
            file_path = os.path.join(folder_path, filename)
            break

    image = read_image(file_path)
    
    # Faire la prédiction
    prediction = model.predict(np.array([image,]))
    
    # Sauvegarder la prédiction
    with open(f"result/{filename}-{datetime.now().timestamp()}.txt", "w") as f:
        f.write(str(prediction))
    
    # Déplacer l'image transformée dans le dossier 'archive'
    shutil.move(file_path, f"archive/{filename}")

    return order[np.argmax([prediction])]