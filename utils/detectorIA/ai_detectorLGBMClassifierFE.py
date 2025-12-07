# Modelo avanzado: TF-IDF + Ingenieria de Caracteristicas Estilometricas (FE) + LightGBM
# ----------------------------------------------------
# ➤ Que hace:
# Este script añade informacion adicional que describe como escribe la persona/modelo:
# TF-IDF (200k features)
# + 7 caracteristicas estilometricas extra (engenieria de features)
# Se combinan con hstack
# LightGBM se entrena con una matriz hibrida

# Paramteros modficados para optimizar velocidad (training pesado)
# ➤ Explicacion tecnica resumida:
# Ademas del contenido textual (TF-IDF), incluye caracteristicas como:
# Longitud media de palabra
# TTR (diversidad lexica)
# Longitud media de oracion
# Densidad de puntuacion
# Frecuencia de palabras funcionales
# Densidad de mayusculas
# Permite detectar señales que no estan en los n-gramas, sino en el estilo.
# Ideal como version avanzada/pro del detector.

#Utiliza el conjunto de desarrollo (DEV_FILE) para detener automaticamente el entrenamiento en el punto optimo, lo que se traduce en una mayor precision y previene el sobreajuste.

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
# Extractor de caracteristicas
STYLOMETRIC_EXTRACTOR_PATH = 'lgbm_stylometric_extractor_fe.pkl'

LABELS = [
    "Completamente escrito por humanos",
    "Iniciado por humanos, continuado por maquina",
    "Escrito por humanos, pulido por maquina",
    "Escrito por maquina, luego humanizado por maquina",
    "Escrito por maquina, luego editado por humanos",
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
# CLASE: EXTRACTOR DE CARACTERiSTICAS ESTILOMeTRICAS
# ----------------------------------------------------
class StylometricFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extrae caracteristicas estilometricas que ayudan a distinguir el estilo.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []
        for text in X:
            # Procesamiento basico
            text_lower = text.lower()
            sentences = re.split(r'[.!?]+\s*', text)
            words = re.findall(r'\b\w+\b', text_lower)
            
            # Evitar division por cero
            num_sentences = len(sentences) if len(sentences) > 0 else 1
            num_words = len(words) if len(words) > 0 else 1
            num_chars = len(text) if len(text) > 0 else 1

            # 1. Longitud promedio de la palabra
            avg_word_len = sum(len(word) for word in words) / num_words
            
            # 2. Diversidad Lexica (Type-Token Ratio)
            ttr = len(set(words)) / num_words
            
            # 3. Longitud promedio de la oracion
            avg_sentence_len = num_words / num_sentences
            
            # 4. Densidad de puntuacion
            punctuation_density = len(re.findall(r'[.,;!?]', text)) / num_words
            
            # 5. Frecuencia de Palabras Funcion Comunes (Ej: 'el', 'que')
            freq_el = words.count('el') / num_words
            freq_que = words.count('que') / num_words
            
            # 6. Densidad de Mayusculas (Uso humano inconsistente)
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
            
        # Devolver una matriz densa de caracteristicas para la concatenacion
        return np.array(features)


def train_and_save_model():
    
    """Entrena el modelo de clasificacion y lo guarda en el disco."""
    print("Iniciando el entrenamiento del modelo de deteccion de IA (LightGBM) con Feature Engineering...")
    
    # --- LoGICA DE NLTK (Verificacion y Descarga) ---
    try:
        # Intentar encontrar el recurso. Si falla, lanza LookupError.
        nltk.data.find('corpora/stopwords')
    # Captura la excepcion base que se lanza (LookupError)
    except LookupError: 
        print("Descargando el recurso 'stopwords' de NLTK...")
        nltk.download('stopwords')
        
    # 1. Definir las stop words originales
    local_stop_words = stopwords.words('spanish')

    # 2. Preprocesar las stop words para consistencia (sin acentos, minusculas)
    preprocessed_stop_words_set = set(
        unidecode.unidecode(word) for word in local_stop_words
    )
    
    # 3. Convertir a LISTA para scikit-learn/LightGBM
    final_stop_words_list = list(preprocessed_stop_words_set)

    # Carga los datos de entrenamiento
    train_df = load_data(TRAIN_FILE)
    
    # --- PASO 1: Vectorizacion TF-IDF ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),    # Trigramas para capturar patrones de IA
        max_features=200000,   # Limite de vocabulario
        lowercase=True, 
        strip_accents='unicode', # Normalizacion de acentos
        min_df=3,              # Ignora terminos muy raros
        stop_words=final_stop_words_list # Lista limpia y consistente
    )

    dev_df = load_data(DEV_FILE)

    X_tfidf = vectorizer.fit_transform(train_df['text']) # Renombrado a X_tfidf
    y_train = train_df['label']

    # 1b. Transformacion TF-IDF para DEV (solo transformacion)
    X_tfidf_dev = vectorizer.transform(dev_df['text']) # 👈 NUEVA LiNEA
    y_dev = dev_df['label'] # 👈 NUEVA LiNEA

     # --- PASO 2: Extraccion de Caracteristicas Estilometricas ---
    stylometric_extractor = StylometricFeatureExtractor()
    X_stylometric = stylometric_extractor.fit_transform(train_df['text'])
    X_stylometric_dev = stylometric_extractor.transform(dev_df['text'])

      # --- PASO 3: Combinar las dos matrices ---
    # Usar hstack para concatenar la matriz dispersa TF-IDF con la matriz densa estilometrica
    X_train_combined = hstack([X_tfidf, X_stylometric])
    X_dev_combined = hstack([X_tfidf_dev, X_stylometric_dev])

    print(f"INFO: Matriz de caracteristicas final creada con {X_train_combined.shape[1]} columnas.") # Seran 200,007 columnas
    
    
    # LightGBM (LGBMClassifier)
    print("INFO: Entrenando el modelo LightGBM...")
    
    classifier = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(LABELS),      
        metric='multi_logloss',     
        random_state=42,            
        n_jobs=-1,                  
        
       
    # ----------------------------------------------------------------------------------
    # --- CONFIGURACIoN DE VELOCIDAD PARA LA MATRIZ DE 200,007 COLUMNAS ---
    # ----------------------------------------------------------------------------------

        n_estimators=1800,           # OPTIMIZACION DE VELOCIDAD: Aumento drastica del numero de arboles (de 1000 a 1800) para mejorar la precision.
        learning_rate=0.05,          # OPTIMIZACION DE VELOCIDAD: Aumenta el paso de aprendizaje para que el modelo converja mas rapido con menos arboles.
        num_leaves=90,               # OPTIMIZACION DE VELOCIDAD: Aumenta la complejidad maxima de cada arbol. Permite capturar señales mas sutiles en las 200,007 caracteristicas.
        min_child_samples=30,        # OPTIMIZACION DE ESTABILIDAD: Asegura que las reglas de division se basen en muestras mas grandes, previniendo el sobreajuste en features dispersas (TF-IDF).
        class_weight='balanced',     # CLAVE PARA EL F1-SCORE: Fuerza al modelo a prestar mayor atencion a las clases minoritarias, como la "Escrito por maquina, luego editado por humanos".
            
        # OPTIMIZACION PARA MATRICES DISPERSAS (TF-IDF)
        sparse_feature=True,        # Indica a LGBM que use optimizaciones para datos dispersos.
        max_bin=63,                 
        verbose=-1                  
    )
    
    # Entrenar con la matriz combinada, usando Early Stopping
    classifier.fit(
    X_train_combined,
    y_train,
    #eval_set=[(X_dev_combined, y_dev)], #  Conjunto de validacion
    #eval_metric='multi_logloss',
    #callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)] # Parar si no mejora en 50 iteraciones
)
    
    # Guarda el vectorizador, el extractor y el modelo para su uso futuro
    joblib.dump(classifier, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(stylometric_extractor, STYLOMETRIC_EXTRACTOR_PATH) 
    
    print("Modelo y vectorizador guardados con exito.")


def predict_ai_content(document_content):
    """
    Carga el modelo guardado, predice la categoria del documento y calcula el AI Score.
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

    # 1. Transformacion TF-IDF
    X_tfidf = vectorizer.transform([document_content])
    
    # 2. Extraccion Estilometrica
    X_stylometric = stylometric_extractor.transform([document_content])
    
    # 3. Combinacion de caracteristicas
    X_combined = hstack([X_tfidf, X_stylometric])
    
    # Realiza la prediccion de probabilidades
    probabilities = classifier.predict_proba(X_combined)[0]
    prediction_index = np.argmax(probabilities)

    # 4. Calculo del AI Score
    ai_score_percent = sum(prob * weight for prob, weight in zip(probabilities, IA_WEIGHTS)) * 100
    
    # 5. Obtener la etiqueta
    if 0 <= prediction_index < len(LABELS):
        prediction_label = LABELS[prediction_index]
    else:
        prediction_label = "Categoria desconocida"

    # 6. Devolver el diccionario
    return {
        "label": prediction_label,    
        "ai_score": ai_score_percent  
    }


def evaluate_model(labels_list):
    """Evalua el modelo con el conjunto de datos de desarrollo y muestra el informe."""
    # Verificar existencia de todos los archivos
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and os.path.exists(STYLOMETRIC_EXTRACTOR_PATH)):
        print("Modelo no entrenado. Por favor, entrene el modelo primero.")
        return
        
    print("Evaluando el modelo  de deteccion de IA (LightGBM) con Feature Engineering...")
    dev_df = load_data(DEV_FILE)
    
    # Carga todos los objetos
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    stylometric_extractor = joblib.load(STYLOMETRIC_EXTRACTOR_PATH) 

    # 1. Transformacion TF-IDF
    X_tfidf = vectorizer.transform(dev_df['text'])
    
    # 2. Extraccion Estilometrica
    X_stylometric = stylometric_extractor.transform(dev_df['text'])
    
    # 3. Combinacion de caracteristicas
    X_dev_combined = hstack([X_tfidf, X_stylometric])
    
    y_dev = dev_df['label']
    
    predictions = classifier.predict(X_dev_combined)

    # 4. Genera el informe de clasificacion
    report_dict = classification_report(y_dev, predictions, target_names=labels_list, output_dict=True)
    report_string = classification_report(y_dev, predictions, target_names=labels_list, output_dict=False)
    
    accuracy_value = report_dict.get('accuracy', 0.0)

    print("\n--- Informe de Clasificacion (Consola) ---")
    print(report_string)
    print("---------------------------------")

    return report_string, accuracy_value

# Entrenar el modelo al iniciar el script por primera vez si no existe
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH) or not os.path.exists(STYLOMETRIC_EXTRACTOR_PATH):
        train_and_save_model()

        evaluate_model(LABELS)