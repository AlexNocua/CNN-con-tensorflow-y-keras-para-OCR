import pandas as pd
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.sax.saxutils import escape
import cv2
from tqdm import tqdm  # comandos rapido de avance
import matplotlib.pyplot as plt
import random as rn
import pickle


# atributos = ['classe']

# creando la tabla de datos
# df = pd.read_csv(dataset_dir, names = atributos)
# funcion para generar los datos
def Generar_datos():
    i = 0
    data = []

    for categoria in CATEGORIAS:
        i += 1
        print(os.path.join(dataset_dir, categoria), i)
        path = os.path.join(dataset_dir, categoria)
        valor = CATEGORIAS.index(categoria)
        listdir = os.listdir(path)
        for i in tqdm(range(len(listdir)), desc=categoria):

            imagen_nombre = listdir[i]
            try:

                imagen_ruta = os.path.join(path, imagen_nombre)
                imagen = cv2.imread(imagen_ruta, cv2.IMREAD_GRAYSCALE)
                imagen = cv2.resize(imagen, (Imagen_size, Imagen_size))
                # plt.imshow(imagen, cmap='gray')
                # plt.show()

                data.append([imagen, valor])

            except Exception as e:
                pass
    rn.shuffle(data)
    x = []
    y = []

    for i in tqdm(range(len(data)), desc="Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    # print(x)
    x = np.array(x).reshape(-1, Imagen_size, Imagen_size, 1)

    pickle_out = open("x_testing.pickle", "wb")
    pickle.dump(x, pickle_out)
    pickle_out.close()

    pickle_out = open("y_testing.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()

    unique_classes_in_y_test = np.unique(y_test)


    # Imprimir las clases únicas presentes en y_test
    print("Clases en y_test:", unique_classes_in_y_test)
    print("Número total de clases en y_test:", len(unique_classes_in_y_test))

CATEGORIAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
              "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
Imagen_size = 80

if __name__ == "__main__":
    # Directorio con las imágenes de entrenamiento
    dataset_dir = 'C:/Users/Alex Nocua/Desktop/Proyectos IA/Standard OCR Datashet/data/testing_data'

    Generar_datos()
