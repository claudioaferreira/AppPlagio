# Clasificación de texto con Machine Learning
# El LightGBM es una técnica de boosting que construye muchos árboles de decisión secuencialmente.
# Captura No Linealidad: Los árboles de decisión pueden capturar relaciones complejas y no lineales entre las combinaciones de trigramas (las features) y las etiquetas de clase, algo que la Regresión Logística no puede hacer. Esto es crucial para distinguir los sutiles matices del texto "humanizado" o "mezclado".
# Rendimiento en Datos Dispersos: LightGBM utiliza un algoritmo llamado GOSS (Gradient-based One-Side Sampling) que le permite entrenar de forma muy rápida y eficiente en conjuntos de datos con muchas features cero (como tu matriz TF-IDF), superando a menudo a XGBoost y a modelos lineales en velocidad y precisión.
# Hiperparámetros de F1-score:
# n_estimators=1000 y learning_rate=0.05: Utilizamos muchos árboles y pasos pequeños para afinar el modelo y evitar el sobreajuste.
# num_leaves=60: Aumentamos la complejidad de cada árbol, permitiendo que el modelo aprenda patrones más intrincados.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
import joblib
import os
import json
import nltk
from nltk.corpus import stopwords
import unidecode
import lightgbm as lgb 


# Define las rutas de los archivos del dataset
TRAIN_FILE = os.path.join('pan-25-ai-detection', 'train.jsonl')
DEV_FILE = os.path.join('pan-25-ai-detection', 'dev.jsonl')
MODEL_PATH = 'lgbm_classifier_modelFIX.pkl'
VECTORIZER_PATH = 'tfidf_lgbm_vectorizerFIX.pkl'

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
    print("Iniciando el entrenamiento del modelo de detección de IA (LightGBM)...")
    
    # --- LÓGICA DE NLTK (Verificación y Descarga) ---
    try:
        # Intentar encontrar el recurso. Si falla, lanza LookupError.
        nltk.data.find('corpora/stopwords')
    # Captura la excepción base que se lanza (LookupError)
    except LookupError: 
        print("Descargando el recurso 'stopwords' de NLTK...")
        nltk.download('stopwords')
        
    # 1. Definir las stop words originales
    local_stop_words = stopwords.words('spanish')

    # 2. Preprocesar las stop words para consistencia (sin acentos, minúsculas)
    preprocessed_stop_words_set = set(
        unidecode.unidecode(word) for word in local_stop_words
    )
    
    # 3. Convertir a LISTA para scikit-learn/LightGBM
    final_stop_words_list = list(preprocessed_stop_words_set)

    # Carga los datos de entrenamiento
    train_df = load_data(TRAIN_FILE)
    
    # Vectoriza el texto usando TF-IDF (CONFIGURACIÓN ÓPTIMA)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),    # Trigramas para capturar patrones de IA
        max_features=200000,   # Límite de vocabulario
        lowercase=True, 
        strip_accents='unicode', # Normalización de acentos
        min_df=3,              # Ignora términos muy raros
        stop_words=final_stop_words_list # Lista limpia y consistente
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']

    print(f"INFO: Matriz de características creada con {X_train.shape[1]} columnas.")
    
    # 🌟 ¡NUEVO CLASIFICADOR! LightGBM (LGBMClassifier)
    print("INFO: Entrenando el modelo LightGBM...")
    
    classifier = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(LABELS),      
        metric='multi_logloss',     
        random_state=42,            
        n_jobs=-1,                  
        
        # AJUSTES DE RENDIMIENTO Y PRECISIÓN
        n_estimators=750,           # Más potencia: Construir más árboles para afinar la decisión, lo que puede ayudar a diferenciar mejor.
        learning_rate=0.08,          # Más precisión: Reducir un poco el paso para asegurar que los 750 árboles sean muy precisos.
        num_leaves=50,              # Más complejidad: Permitir que los árboles modelen interacciones más complejas, clave para texto "humanizado."
        min_child_samples=30,       # Más sensibilidad: Reducir este valor le permite al modelo crear reglas de clasificación basadas en grupos más pequeños, mejorando el recall en clases minoritarias/difíciles.
        class_weight='balanced',    # Fuerza a LGBM a considerar el desbalance.
        
        # OPTIMIZACIÓN PARA MATRICES DISPERSAS (TF-IDF)
        sparse_feature=True,        # Indica a LGBM que use optimizaciones para datos dispersos.
        max_bin=63,                 
        verbose=-1                  
    )
    
    classifier.fit(X_train, y_train)
    
    # Guarda el vectorizador y el modelo para su uso futuro
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    
    print("Modelo y vectorizador guardados con éxito.")


def predict_ai_content(document_content):
    """Carga el modelo guardado y predice la categoría del documento."""
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
        return LABELS[prediction_number]
    else:
        return "Categoría desconocida"

def evaluate_model(labels_list):
    """Evalúa el modelo con el conjunto de datos de desarrollo y muestra el informe. - MODELO TF-IDF + LightGBM"""
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

    report_dict = classification_report(y_dev, predictions, target_names=labels_list, output_dict=True)
    report_string = classification_report(y_dev, predictions, target_names=labels_list, output_dict=False)
    
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