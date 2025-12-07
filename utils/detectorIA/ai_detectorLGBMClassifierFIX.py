# Modelo estandar basado unicamente en TF-IDF + LightGBM

# ➤ Este script usa un pipeline clasico:
# TF-IDF (1–3 n-gramas)
# Entrena LightGBM como clasificador multicategoria
# No usa inforacion estilométrica extra
# Usa parametros optimizados para F1-score
# Produce no solo la categoria, sino un puntaje IA (IA-score) basado en pesos
# ➤ Explicacion técnica resumida:
# El modelo se basa exclusivamente en la representacion vectorial TF-IDF.
# Todas las features provienen del texto segmentado en tokens/gramas.
# No evalua estilo, solo contenido.
# Es mas rapido y mas simple.
# Ideal como baseline o como version “lite”.
# ----------------------------------------------------

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
import numpy as np

# Define las rutas de los archivos del dataset
TRAIN_FILE = os.path.join('pan-25-ai-detection', 'train.jsonl')
DEV_FILE = os.path.join('pan-25-ai-detection', 'dev.jsonl')
MODEL_PATH = 'lgbm_classifier_modelFIX.pkl'
VECTORIZER_PATH = 'tfidf_lgbm_vectorizerFIX.pkl'

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


def train_and_save_model():
    
    """Entrena el modelo de clasificacion y lo guarda en el disco."""
    print("Iniciando el entrenamiento del modelo de deteccion de IA (LightGBM)...")
    
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
    
    # Vectoriza el texto usando TF-IDF (CONFIGURACIoN oPTIMA)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),    # Trigramas para capturar patrones de IA
        max_features=200000,   # Limite de vocabulario
        lowercase=True, 
        strip_accents='unicode', # Normalizacion de acentos
        min_df=3,              # Ignora términos muy raros
        stop_words=final_stop_words_list # Lista limpia y consistente
    )
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']

    print(f"INFO: Matriz de caracteristicas creada con {X_train.shape[1]} columnas.")
    
    # 🌟 ¡NUEVO CLASIFICADOR! LightGBM (LGBMClassifier)
    print("INFO: Entrenando el modelo LightGBM...")
    
    classifier = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(LABELS),      
        metric='multi_logloss',     
        random_state=42,            
        n_jobs=-1,                  
        
        # AJUSTES DE RENDIMIENTO Y PRECISIoN
        n_estimators=750,           # Mas potencia: Construir mas arboles para afinar la decision, lo que puede ayudar a diferenciar mejor.
        learning_rate=0.08,          # Mas precision: Reducir un poco el paso para asegurar que los 750 arboles sean muy precisos.
        num_leaves=50,              # Mas complejidad: Permitir que los arboles modelen interacciones mas complejas, clave para texto "humanizado."
        min_child_samples=30,       # Mas sensibilidad: Reducir este valor le permite al modelo crear reglas de clasificacion basadas en grupos mas pequeños, mejorando el recall en clases minoritarias/dificiles.
        class_weight='balanced',    # Fuerza a LGBM a considerar el desbalance.
        
        # OPTIMIZACIoN PARA MATRICES DISPERSAS (TF-IDF)
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
    """Carga el modelo guardado y predice la categoria del documento."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("El modelo no existe. Entrenando primero...")
        train_and_save_model()
        
    # Carga el modelo y el vectorizador
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Transforma el contenido del nuevo documento
    text_vectorized = vectorizer.transform([document_content])

    # 1. Obtenemos las probabilidades para TODAS las 6 clases
    probabilities = classifier.predict_proba(text_vectorized)[0]
    
    # 2. Obtenemos la prediccion principal (la clase con mayor probabilidad)
    prediction_index = np.argmax(probabilities)

    # 3. Calculamos el "Puntaje IA"
    #    Asumimos que la clase 0 ("Completamente escrito por humanos") es la unica "humana".
    #    El resto de clases (1 a 5) implican algun nivel de IA.
    #human_score_prob = probabilities[0]

    # El puntaje de IA es la suma de las probabilidades del resto de clases.
    # Multiplicamos por 100 para tener el porcentaje.
    ai_score_percent = sum(prob * weight for prob, weight in zip(probabilities, IA_WEIGHTS)) * 100

    # 4. Obtenemos la etiqueta de texto de la prediccion principal
    if 0 <= prediction_index < len(LABELS):
        prediction_label = LABELS[prediction_index]
    else:
        prediction_label = "Categoria desconocida"

        # 5. Devolvemos un diccionario con toda la informacion
    return {
        "label": prediction_label,    # Ej: "Iniciado por humanos..."
        "ai_score": ai_score_percent  # Ej: 96.36
    }

def get_text_stats(document_content):
    """Calcula estadisticas simples del texto."""
    characters = len(document_content)
    words = len(document_content.split())
    return {
        "characters": characters,
        "words": words
    }

def evaluate_model(labels_list):
    """Evalua el modelo con el conjunto de datos de desarrollo y muestra el informe. - MODELO TF-IDF + LightGBM"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("Modelo no entrenado. Por favor, entrene el modelo primero.")
        return
        
    print("Evaluando el modelo de deteccion de IA TF-IDF + LightGBM")
    dev_df = load_data(DEV_FILE)
    
    classifier = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    X_dev = vectorizer.transform(dev_df['text'])
    y_dev = dev_df['label']
    
    predictions = classifier.predict(X_dev)

    report_dict = classification_report(y_dev, predictions, target_names=labels_list, output_dict=True)
    report_string = classification_report(y_dev, predictions, target_names=labels_list, output_dict=False)
    
    accuracy_value = report_dict.get('accuracy', 0.0)

    print("\n--- Informe de Clasificacion (Consola) ---")
    print(report_string)
    print("---------------------------------")

    return report_string, accuracy_value

# Entrenar el modelo al iniciar el script por primera vez si no existe
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        train_and_save_model()

        evaluate_model(LABELS)