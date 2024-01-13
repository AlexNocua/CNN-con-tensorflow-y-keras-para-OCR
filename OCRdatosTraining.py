import os
import cv2
import numpy as np
import random as rn
from tqdm import tqdm
import Augmentor
import pickle

def Generar_datos():
    i = 0
    data = []

    for categoria in CATEGORIAS:
        i += 1
        print(os.path.join(dataset_dir, categoria),i)
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
    x =[]
    y= []
    

    for i in tqdm(range(len(data)), desc="Procesamiento"):
        par = data[i]
        x.append(par[0])
        y.append(par[1])

    x = np.array(x).reshape(-1,Imagen_size,Imagen_size,1)

    if len(x) > 0:
        pickle_out = open ("x_Training.pickle","wb")
        pickle.dump(x, pickle_out)
        pickle_out.close()
        print("Datos guardados en x_Training.")
    else:
        print("No se encontraron datos para guardar en x_test.pickle")

    pickle_out = open ("y_Training.pickle","wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()
    print("Datos guardados en x_Training.pickle")

CATEGORIAS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A", "B", "C", "D",
              "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S","T","U","W","X","Y","Z"]
Imagen_size = 80

if __name__ == "__main__":
    dataset_dir = 'C:/Users/Alex Nocua/Desktop/Proyectos IA/Standard OCR Datashet/data/training_data'
    Generar_datos()
