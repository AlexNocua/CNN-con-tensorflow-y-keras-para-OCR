import pandas as pd
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

# Directorio con las imágenes de entrenamiento
dataset_dir = 'C:/Users/Alex Nocua/Desktop/Proyectos IA/Standard OCR Datashet/data/testing_data'


# Cargar imágenes utilizando ImageDataGenerator
image_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normaliza los valores de píxeles a valores entre 0 y 1
    rotation_range=10,     # Rotación aleatoria de la imagen
    width_shift_range=0.1, # Desplazamiento horizontal aleatorio
    height_shift_range=0.1,# Desplazamiento vertical aleatorio
    shear_range=0.1,       # Cambio de inclinación
    zoom_range=0.1,        # Zoom aleatorio
    horizontal_flip=True,  # Volteo horizontal aleatorio
    fill_mode='nearest'    # Relleno de píxeles
)

# Crea un generador de datos de imágenes a partir del directorio de imágenes
image_generator = image_datagen.flow_from_directory(
    dataset_dir,
    target_size=(150, 150),  # Tamaño de las imágenes de entrada
    batch_size=32,           # Tamaño del lote
    class_mode='categorical' # Modo de clasificación categórica
)

NEURONIOS_OCULTOS = 400
NEURONIOS_SAIDA = 36   #Asegúrate de que esto coincida con el número de clases

# Definir el modelo MLP
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.1), #
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),#
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.1),
    Flatten(),#
    Dense(NEURONIOS_OCULTOS, activation="relu"),
    Dense(NEURONIOS_SAIDA, activation="softmax")
])

# Compilar el modelo
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(
    image_generator,
    steps_per_epoch=image_generator.samples // 64,
    epochs=45
)

#XyY
y_pred = model.predict(image_generator)
y_pred = np.argmax(y_pred, axis=1)

#rotulos verdaderos
y_true = image_generator.classes

#Matriz de Confusão
conf_mat = tf.math.confusion_matrix(y_true, y_pred).numpy()
print(conf_mat)

#precicion

recall_metric = tf.keras.metrics.Recall()
recall_metric.update_state(y_true, y_pred)
recall = recall_metric.result().numpy()

precision_metric = tf.keras.metrics.Precision()
precision_metric.update_state(y_true, y_pred)
precision = precision_metric.result().numpy()

#F1
f1_score = 2 * (precision * recall) / (precision + recall)

# Crear un documento XML para el historial de entrenamiento
root = ET.Element("1Entrenamiento")
accuracy_list= []
loss_list = []

for epoch, (acc, loss) in enumerate(zip(history.history['accuracy'], history.history['loss'])): 
    # Acurácia e perda
    epoch_element = ET.Element("epoch", number=str(epoch + 1))
    accuracy_element = ET.Element("accuracy")
    accuracy_element.text = str(acc)
    loss_element = ET.Element("loss")
    loss_element.text = str(loss)
    
    # Precisão
    precision_element = ET.Element("precision")
    precision_element.text = str(precision)
    
    # Recall
    recall_element = ET.Element("recall")
    recall_element.text = str(recall)
    
    # F1-Score
    f1_score_element = ET.Element("f1_score")
    f1_score_element.text = str(f1_score)
      
    # Matriz de confusão
    #conf_mat_element = ET.Element("confusion_matrix")
    #conf_mat_element.text = str(conf_mat.tolist()) 

    accuracy_list.append(acc)
    loss_list.append(loss)

    # Adicionar todos os elementos à época
    epoch_element.append(accuracy_element)
    epoch_element.append(loss_element)
    epoch_element.append(precision_element)
    epoch_element.append(recall_element)
    epoch_element.append(f1_score_element)
    #epoch_element.append(conf_mat_element)
    
    root.append(epoch_element)

# Creamos un DataFrame
df = pd.DataFrame({
    'Accuracy': accuracy_list,
    'Aoss': loss_list,
    })

# guardamos el DataFrame en un archivo de Excel
df.to_excel('1_training_data.xlsx', index=False) 

# Criar uma árvore a partir do elemento raiz
tree = ET.ElementTree(root)

# Escrever a árvore em um arquivo XML
tree.write("3Entrenamiento.xml")

# Formatear el archivo XML para mejor legibilidad
xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="")
with open("1Entrenamiento.xml", "w") as xml_file:
    xml_file.write(xmlstr)