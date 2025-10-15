import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
import io

## 1 Carga de datos (load_data)
# Lee los archivos .jsonl de entrenamiento (train.jsonl) y validación (dev.jsonl), los transforma en un DataFrame con columnas text y label.

## 2 Vectorización con TF-IDF (TfidfVectorizer)
# Vectorización con TF-IDF (TfidfVectorizer) y clasificación con Regresión Logística
# Convierte los textos en una matriz dispersa de frecuencias ponderadas de términos (clásico bag-of-words con TF-IDF)
# Ejemplo: "El perro ladra" → [0.23, 0.0, 0.89, ...]

## 3 Entrenamiento de un clasificador (LogisticRegression)
# Usa regresión logística multiclase (max_iter=500) sobre los vectores TF-IDF para aprender a predecir la etiqueta.
# Guarda el modelo entrenado (.pkl) y el vectorizador para reuso.

## 4 Predicción (predict_ai_content)
# Carga el modelo y el vectorizador.
# Transforma un nuevo documento a TF-IDF.
# El clasificador devuelve un número de clase → lo traduces a texto usando LABELS.

## 5 Evaluación (evaluate_model)
# Evalúa en el conjunto dev.jsonl y muestra métricas (classification_report).


# Define las rutas de los archivos del dataset
TRAIN_FILE = os.path.join('pan-25-ai-detection', 'train.jsonl')
DEV_FILE = os.path.join('pan-25-ai-detection', 'dev.jsonl')
MODEL_PATH = 'ai_classifier_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

LABELS = [
    "Fully human-written",
    "Human-initiated, then machine-continued",
    "Human-written, then machine-polished",
    "Machine-written, then machine-humanized",
    "Machine-written, then human-edited",
    "Deeply-mixed text"
]

def load_data(file_path):
    """Carga los datos de un archivo .jsonl y los devuelve como un DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(eval(line.strip()))
    return pd.DataFrame(data)

def train_and_save_model():
    """Entrena el modelo de clasificación y lo guarda en el disco."""
    print("Iniciando el entrenamiento del modelo de detección de IA...")
    
    # Carga los datos de entrenamiento
    train_df = load_data(TRAIN_FILE)
    
    # Vectoriza el texto usando TF-IDF
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    
    # Entrena el modelo de regresión logística
    classifier = LogisticRegression(max_iter=500)
    classifier.fit(X_train, y_train)
    
    # Guarda el vectorizador y el modelo para su uso futuro
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Modelo y vectorizador guardados con éxito.")

def predict_ai_content(document_content):
    """Carga el modelo guardado y predice la categoría del documento."""
    # Si el modelo no existe, lo entrena primero
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("El modelo no existe. Entrenando primero...")
        train_and_save_model()
    
    # Carga el modelo y el vectorizador desde el disco
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Transforma el contenido del nuevo documento
    text_vectorized = vectorizer.transform([document_content])
    
    # Realiza la predicción
    prediction = classifier.predict(text_vectorized)
    
    # La predicción te da un número, necesitas convertirlo en una etiqueta de texto
    prediction_number = prediction[0]



    # Verifica si el número está dentro del rango de etiquetas
    if 0 <= prediction_number < len(LABELS):
        # Devuelve la etiqueta de texto en lugar del número
        return LABELS[prediction_number]
    else:
        # En caso de un error inesperado, devuelve un mensaje
        return "Categoría desconocida"

def evaluate_model(labels_list):
    """Evalúa el modelo con el conjunto de datos de desarrollo y muestra el informe."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Modelo no entrenado. Por favor, entrene el modelo primero.")
        return
        
    print("Evaluando el modelo con el conjunto de datos de desarrollo...")
    dev_df = load_data(DEV_FILE)
    
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    X_dev = vectorizer.transform(dev_df['text'])
    y_dev = dev_df['label']
    
    predictions = classifier.predict(X_dev)
    report_string = classification_report(y_dev, predictions, target_names=labels_list)
    
    print("\n--- Informe de Clasificación (Consola) ---")
    print(report_string)
    print("---------------------------------")
    return report_string

# Entrenar el modelo al iniciar el script por primera vez si no existe
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_and_save_model()

        evaluate_model()