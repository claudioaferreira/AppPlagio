# Modelo avanzado: TF-IDF + Ingeniería de Características Estilométricas (FE) + LightGBM
# El modelo más avanzado para la detección de IA en textos.
# ➤ Qué hace:
# Este script añade información adicional que describe cómo escribe la persona/modelo:
# TF-IDF (200k features)
# + 7 características estilométricas extra (engeniería de features)
# Se combinan con hstack
# LightGBM se entrena con una matriz híbrida

# Parámteros modficados para optimizar velocidad (training pesado)
# ➤ Explicación técnica resumida:
# Además del contenido textual (TF-IDF), incluye características como:
# Longitud media de palabra
# TTR (diversidad léxica)
# Longitud media de oración
# Densidad de puntuación
# Frecuencia de palabras funcionales
# Densidad de mayúsculas
# Permite detectar señales que no están en los n-gramas, sino en el estilo.
# Es un modelo más completo, robusto y difícil de engañar.
# Ideal como versión avanzada/pro del detector.

#Utiliza el conjunto de desarrollo (DEV_FILE) para detener automáticamente el entrenamiento en el punto óptimo, lo que se traduce en una mayor precisión y previene el sobreajuste.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack 
import numpy as np 
import re 
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

MODEL_PATH = 'lgbm_classifier_model_fe.pkl' 
# El vectorizador TF-IDF usado con el modelo FE
VECTORIZER_PATH = 'tfidf_lgbm_vectorizer_fe.pkl' 
# El nuevo extractor de características
STYLOMETRIC_EXTRACTOR_PATH = 'lgbm_stylometric_extractor_fe.pkl'

LABELS = [
    "Completamente escrito por humanos",
    "Iniciado por humanos, continuado por máquina",
    "Escrito por humanos, pulido por máquina",
    "Escrito por máquina, luego humanizado por máquina",
    "Escrito por máquina, luego editado por humanos",
    "Texto profundamente mezclado"
]

IA_WEIGHTS = [0.0, 0.5, 0.4, 0.8, 0.6, 1.0]

def load_data(file_path):
    """Carga los datos de un archivo .jsonl y los devuelve como un DataFrame."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return pd.DataFrame(data)


# ----------------------------------------------------
# CLASE: EXTRACTOR DE CARACTERÍSTICAS ESTILOMÉTRICAS
# ----------------------------------------------------
class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae características estilométricas que ayudan a distinguir el estilo.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            # Procesamiento básico
            text_lower = text.lower()
            sentences = re.split(r'[.!?]+\s*', text)
            words = re.findall(r'\b\w+\b', text_lower)
            
            # Evitar división por cero
            num_sentences = len(sentences) if len(sentences) > 0 else 1
            num_words = len(words) if len(words) > 0 else 1
            num_chars = len(text) if len(text) > 0 else 1

            # 1. Longitud promedio de la palabra
            avg_word_len = sum(len(word) for word in words) / num_words
            
            # 2. Diversidad Léxica (Type-Token Ratio)
            ttr = len(set(words)) / num_words
            
            # 3. Longitud promedio de la oración
            avg_sentence_len = num_words / num_sentences
            
            # 4. Densidad de puntuación
            punctuation_density = len(re.findall(r'[.,;!?]', text)) / num_words
            
            # 5. Frecuencia de Palabras Función Comunes (Ej: 'el', 'que')
            freq_el = words.count('el') / num_words
            freq_que = words.count('que') / num_words
            
            # 6. Densidad de Mayúsculas (Uso humano inconsistente)
            upper_density = sum(1 for char in text if char.isupper()) / num_chars

            features.append([
                avg_word_len,
                ttr,
                avg_sentence_len,
                punctuation_density,
                freq_el,
                freq_que,
                upper_density
            ])
            
        # Devolver una matriz densa de características para la concatenación
        return np.array(features)


def train_and_save_model():
    
    """Entrena el modelo de clasificación y lo guarda en el disco."""
    print("Iniciando el entrenamiento del modelo de detección de IA (LightGBM) con Feature Engineering...")
    
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
    
    # --- PASO 1: Vectorización TF-IDF ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),    # Trigramas para capturar patrones de IA
        max_features=200000,   # Límite de vocabulario
        lowercase=True, 
        strip_accents='unicode', # Normalización de acentos
        min_df=3,              # Ignora términos muy raros
        stop_words=final_stop_words_list # Lista limpia y consistente
    )

    dev_df = load_data(DEV_FILE)

    X_tfidf = vectorizer.fit_transform(train_df['text']) # Renombrado a X_tfidf
    y_train = train_df['label']

    # 1b. Transformación TF-IDF para DEV (solo transformación)
    X_tfidf_dev = vectorizer.transform(dev_df['text']) # 👈 NUEVA LÍNEA
    y_dev = dev_df['label'] # 👈 NUEVA LÍNEA

     # --- PASO 2: Extracción de Características Estilométricas ---
    stylometric_extractor = StylometricFeatureExtractor()
    X_stylometric = stylometric_extractor.fit_transform(train_df['text'])
    X_stylometric_dev = stylometric_extractor.transform(dev_df['text'])

      # --- PASO 3: Combinar las dos matrices ---
    # Usar hstack para concatenar la matriz dispersa TF-IDF con la matriz densa estilométrica
    X_train_combined = hstack([X_tfidf, X_stylometric])
    X_dev_combined = hstack([X_tfidf_dev, X_stylometric_dev])

    print(f"INFO: Matriz de características final creada con {X_train_combined.shape[1]} columnas.") # Serán 200,007 columnas
    
    
    # ¡NUEVO CLASIFICADOR! LightGBM (LGBMClassifier)
    print("INFO: Entrenando el modelo LightGBM...")
    
    classifier = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(LABELS),      
        metric='multi_logloss',     
        random_state=42,            
        n_jobs=-1,                  
        
       
    # ----------------------------------------------------------------------------------
    # --- CONFIGURACIÓN DE VELOCIDAD EXTREMA PARA LA MATRIZ DE 200,007 COLUMNAS ---
    # ----------------------------------------------------------------------------------

        n_estimators=1800,           # OPTIMIZACIÓN DE VELOCIDAD: Aumento drástica del número de árboles (de 1000 a 1800) para mejorar la precisión.
        learning_rate=0.05,          # COMPENSACIÓN DE VELOCIDAD: Aumenta el paso de aprendizaje para que el modelo converja más rápido con menos árboles.
        num_leaves=90,               # OPTIMIZACIÓN DE VELOCIDAD: Aumenta la complejidad máxima de cada árbol. Permite capturar señales más sutiles en las 200,007 características.
        min_child_samples=30,        # OPTIMIZACIÓN DE ESTABILIDAD: Asegura que las reglas de división se basen en muestras más grandes, previniendo el sobreajuste en features dispersas (TF-IDF).
        class_weight='balanced',     # CLAVE PARA EL F1-SCORE: Fuerza al modelo a prestar mayor atención a las clases minoritarias, como la "Escrito por máquina, luego editado por humanos".
            
        # OPTIMIZACIÓN PARA MATRICES DISPERSAS (TF-IDF)
        sparse_feature=True,        # Indica a LGBM que use optimizaciones para datos dispersos.
        max_bin=63,                 
        verbose=-1                  
    )
    
    # Entrenar con la matriz combinada, usando Early Stopping
    classifier.fit(
    X_train_combined,
    y_train,
    #eval_set=[(X_dev_combined, y_dev)], #  Conjunto de validación
    #eval_metric='multi_logloss',
    #callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)] # Parar si no mejora en 50 iteraciones
)
    
    # Guarda el vectorizador, el extractor y el modelo para su uso futuro
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(stylometric_extractor, STYLOMETRIC_EXTRACTOR_PATH) 
    
    print("Modelo y vectorizador guardados con éxito.")


def predict_ai_content(document_content):
    """
    Carga el modelo guardado, predice la categoría del documento y calcula el AI Score.
    Retorna un diccionario: {"label": "...", "ai_score": ...}
    """
    # Verificar existencia de todos los archivos
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(STYLOMETRIC_EXTRACTOR_PATH)):
        print("El modelo no existe. Entrenando primero...")
        train_and_save_model()
        
    # Carga todos los objetos desde el disco
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    stylometric_extractor = joblib.load(STYLOMETRIC_EXTRACTOR_PATH) 

    # 1. Transformación TF-IDF
    X_tfidf = vectorizer.transform([document_content])
    
    # 2. Extracción Estilométrica
    X_stylometric = stylometric_extractor.transform([document_content])
    
    # 3. Combinación de características
    X_combined = hstack([X_tfidf, X_stylometric])
    
    # Realiza la predicción de probabilidades
    probabilities = classifier.predict_proba(X_combined)[0]
    prediction_index = np.argmax(probabilities)

    # 4. Cálculo del AI Score
    ai_score_percent = sum(prob * weight for prob, weight in zip(probabilities, IA_WEIGHTS)) * 100
    
    # 5. Obtener la etiqueta
    if 0 <= prediction_index < len(LABELS):
        prediction_label = LABELS[prediction_index]
    else:
        prediction_label = "Categoría desconocida"

    # 6. Devolver el diccionario (soluciona el error jinja2)
    return {
        "label": prediction_label,    
        "ai_score": ai_score_percent  
    }


def evaluate_model(labels_list):
    """Evalúa el modelo con el conjunto de datos de desarrollo y muestra el informe."""
    # Verificar existencia de todos los archivos
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(STYLOMETRIC_EXTRACTOR_PATH)):
        print("Modelo no entrenado. Por favor, entrene el modelo primero.")
        return
        
    print("Evaluando el modelo  de detección de IA (LightGBM) con Feature Engineering...")
    dev_df = load_data(DEV_FILE)
    
    # Carga todos los objetos
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    stylometric_extractor = joblib.load(STYLOMETRIC_EXTRACTOR_PATH) 

    # 1. Transformación TF-IDF
    X_tfidf = vectorizer.transform(dev_df['text'])
    
    # 2. Extracción Estilométrica
    X_stylometric = stylometric_extractor.transform(dev_df['text'])
    
    # 3. Combinación de características
    X_dev_combined = hstack([X_tfidf, X_stylometric])
    
    y_dev = dev_df['label']
    
    predictions = classifier.predict(X_dev_combined)

    # ... (El resto del código de evaluación se mantiene)
    report_dict = classification_report(y_dev, predictions, target_names=labels_list, output_dict=True)
    report_string = classification_report(y_dev, predictions, target_names=labels_list, output_dict=False)
    
    accuracy_value = report_dict.get('accuracy', 0.0)

    print("\n--- Informe de Clasificación (Consola) ---")
    print(report_string)
    print("---------------------------------")

    return report_string, accuracy_value

# Entrenar el modelo al iniciar el script por primera vez si no existe
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(STYLOMETRIC_EXTRACTOR_PATH):
        train_and_save_model()

        evaluate_model(LABELS)