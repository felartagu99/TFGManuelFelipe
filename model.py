import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import numpy as np
from PIL import Image

# Paso 1: Recopilación de datos y preprocesamiento

# Definir el directorio que contiene las imágenes y las máscaras
data_dir = 'ruta/a/tu/directorio'
classes = ['papel', 'orgánico', 'vidrio', 'plástico']

# Función para cargar las imágenes y preprocesarlas
def load_images_and_labels(data_dir, classes):
    images = []
    labels = []
    for i, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).resize((224, 224))  # Redimensionar la imagen
            img = np.array(img) / 255.0  # Normalizar los valores de píxeles
            images.append(img)
            labels.append(i)  # Etiqueta de la clase
    return np.array(images), np.array(labels)

# Cargar imágenes y etiquetas
images, labels = load_images_and_labels(data_dir, classes)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# Paso 2: Construcción del modelo

# Definir el modelo de red neuronal convolucional
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')  # Capa de salida con activación softmax para clasificación multiclase
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Función de pérdida para clasificación multiclase
              metrics=['accuracy'])

# Paso 3: Entrenamiento del modelo

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Paso 4: Evaluación del modelo

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Precisión en el conjunto de prueba: {test_acc}')

# Guardar el modelo entrenado
model.save('modelo_entrenado.h5')
