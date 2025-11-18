#Clasificación de texto con Machine Learning#
## 1 Carga de datos (load_data)
# Lee los archivos .jsonl de entrenamiento (train.jsonl) y validación (dev.jsonl), los transforma en un DataFrame con columnas text y label.

## 2 Vectorización con TF-IDF (TfidfVectorizer)
# Vectorización con TF-IDF (TfidfVectorizer) y clasificación con Regresión Logística
# Convierte los textos en una matriz dispersa de frecuencias ponderadas de términos (clásico bag-of-words con TF-IDF)
# Ejemplo: "El perro ladra" → [0.23, 0.0, 0.89, ...]
#La Regresión Logística, incluso con regularización, asume una relación lineal entre las features (TF-IDF) y las etiquetas. El texto generado por IA y el texto mezclado a menudo tienen patrones no lineales y complejos.

## 3 Entrenamiento de un clasificador (LogisticRegression)
# Usa regresión logística multiclase (max_iter=500) sobre los vectores TF-IDF para aprender a predecir la etiqueta.
# Guarda el modelo entrenado (.pkl) y el vectorizador para reuso.

## 4 Predicción (predict_ai_content)
# Carga el modelo y el vectorizador.
# Transforma un nuevo documento a TF-IDF.
# El clasificador devuelve un número de clase → lo traduces a texto usando LABELS.

## 5 Evaluación (evaluate_model)
# Evalúa en el conjunto dev.jsonl y muestra métricas (classification_report).

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
import io
import unidecode
import json
import nltk
from nltk.corpus import stopwords


# Define las rutas de los archivos del dataset
TRAIN_FILE = os.path.join('pan-25-ai-detection', 'train.jsonl')
DEV_FILE = os.path.join('pan-25-ai-detection', 'dev.jsonl')
MODEL_PATH = 'ai_classifier_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

LABELS = [
    "Completamente escrito por humanos",
    "Iniciado por humanos, continuado por máquina",
    "Escrito por humanos, pulido por máquina",
    "Escrito por máquina, luego humanizado por máquina",
    "Escrito por máquina, luego editado por humanos",
    "Texto profundamente mezclado"
]

def load_data(file_path):
    """Carga los datos de un archivo .jsonl y los devuelve como un DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
             data.append(json.loads(line.strip()))
    return pd.DataFrame(data)





def train_and_save_model():
    
    """Entrena el modelo de clasificación y lo guarda en el disco."""
    print("Iniciando el entrenamiento del modelo de detección de IA...")
    
     # LÓGICA DE NLTK INCORPORADA: Verifica y descarga 'stopwords' si es necesario.
     ## Esto es crucial para evitar errores en entornos nuevos.
     ## Si ya tienes 'stopwords', no hará nada.
     ## Si no lo tienes, lo descargará automáticamente.
     ## scikit-learn puede usar estas stopwords para filtrar palabras comunes en español.
     ## Esto mejora la calidad del modelo al reducir el ruido.

    try:
        # Intentar encontrar el recurso. Si falla, lanza LookupError.
        nltk.data.find('corpora/stopwords')
    # ¡CORRECCIÓN! Captura la excepción base que se lanza (LookupError)
    except LookupError: 
        print("Descargando el recurso 'stopwords' de NLTK...")
        nltk.download('stopwords')
        

   # Definir las stop words AQUI, después de la descarga confirmada.
    # Usamos una variable local 'local_stop_words'
    local_stop_words = stopwords.words('spanish')

    #  Preprocesar las stop words para consistencia (sin acentos, minúsculas)
    preprocessed_stop_words_set = set(
        unidecode.unidecode(word) for word in local_stop_words
    )

    # Convertir el set a LISTA para scikit-learn.
    final_stop_words_list = list(preprocessed_stop_words_set)

    # Carga los datos de entrenamiento
    train_df = load_data(TRAIN_FILE)
    
    # Vectoriza el texto usando TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),  #Aumentamos a trigramas (1, 3) para detectar patrones más largos y sutiles de la IA.
        max_features=200000, #Limita el vocabulario a 200,000 (ajustable) para controlar el tamaño y la complejidad
        lowercase=True, #Convierte todo a minúsculas para reducir la dimensionalidad
        strip_accents='unicode', #Elimina acentos para normalizar
        min_df=3,  #Ignora términos que aparecen en menos de 3 documentos (reduce ruido)
        stop_words=final_stop_words_list   #Usar la variable preprocesada para que coincida con la configuración. Reduce ruido y enfoca el modelo. variable local que contiene la lista de NLTK
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']

    print(f"INFO: Matriz de características creada con {X_train.shape[1]} columnas.")
    
    # Entrena el modelo de regresión logística
    classifier = LogisticRegression(
        max_iter=5000, # Se puso 5000 porque con 1000 no convergía bien
        class_weight='balanced', 
        solver='saga', #Solucionador más eficiente para datos dispersos y grandes.
        penalty='l1', #L1 para promover la sparsity y reducir overfitting,
        C=5.0,        # ¡Aumentar C (de 0.1 a 5.0) para reducir la regularización y mejorar el aprendizaje.
        n_jobs=-1 #Utiliza todos los núcleos de CPU disponibles para paralelizar (acelerar)
    )
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
    """Evalúa el modelo con el conjunto de datos de desarrollo y muestra el informe.-MODELO TF-IDF + REGRESIÓN LOGÍSTICA"""
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
    # 1. Obtiene el reporte como DICCIONARIO para extraer la precisión
    report_dict = classification_report(y_dev, predictions, target_names=labels_list, output_dict=True)
    
    # 2. Obtiene el reporte como STRING para mostrarlo en la plantilla
    report_string = classification_report(y_dev, predictions, target_names=labels_list, output_dict=False)
    
    # Extrae la precisión total del modelo (accuracy)
    accuracy_value = report_dict.get('accuracy', 0.0)

    print("\n--- Informe de Clasificación (Consola) ---")
    print(report_string)
    print("---------------------------------")

    return report_string, accuracy_value

# Entrenar el modelo al iniciar el script por primera vez si no existe
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_and_save_model()

        evaluate_model(LABELS)