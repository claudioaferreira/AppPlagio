import pyodbc
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import CNXN_STR 

# --- Load the embedding model (XLM-RoBERTa-based Multilingual) ---
# Este es el modelo XLM-RoBERTa (basado en RoBERTa de Facebook) finetuneado 
# en tareas de similitud textual para un rendimiento superior al MPNet en ciertas métricas.
model = SentenceTransformer('stsb-xlm-r-multilingual')

# Definición de la clasificación de riesgo basada en el umbrales de similitud
PLAGIARISM_RISK_THRESHOLDS = {
    'MUY ALTO': 0.85,
    'ALTO': 0.70,
    'MODERADO': 0.50,
    'BAJO': 0.40,
}

def determine_risk_level(score):
    """Clasifica el puntaje de similitud en un nivel de riesgo."""
    if score >= PLAGIARISM_RISK_THRESHOLDS['MUY ALTO']:
        return "MUY ALTO", "risk-muy-alto"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['ALTO']:
        return "ALTO", "risk-alto"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['MODERADO']:
        return "MODERADO", "risk-moderado"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['BAJO']:
        return "BAJO", "risk-bajo"
    else:
        return "INSIGNIFICANTE", "risk-insignificante"


def find_plagiarism_with_roberta_embeddings(document_content):
    """
    Detecta plagio comparando cada frase del nuevo documento 
    contra todos los embeddings del corpus usando el modelo XLM-RoBERTa (granularidad por frase).
    """
    
    # 1. Segmentación del nuevo documento en frases
    try:
        sentences = sent_tokenize(document_content)
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            sentences = sent_tokenize(document_content)
        except:
            sentences = document_content.split('\n')
            sentences = [s.strip() for s in sentences if s.strip()]


    if not sentences:
        return {
            'max_similarity': 0.0,
            'match_count': 0,
            'risk_level': 'SIN TEXTO',
            'risk_color': 'gray',
            'detailed_results': []
        }
    
    # 2. Generar embeddings para cada frase del nuevo documento
    new_embeddings = model.encode(sentences)
    print(f"DEBUG: Dimensión del nuevo Embedding Modelo 3: XLM-RoBERTa: {new_embeddings.shape}") 
    # 3. Conexión a la base de datos y carga de embeddings
    with pyodbc.connect(CNXN_STR) as cnxn:
        cursor = cnxn.cursor()
        cursor.execute("SELECT DocumentID, Embedding FROM DocumentEmbeddings_XLM_R")
        
        db_embeddings = []
        db_ids = []
        for row in cursor.fetchall():
            db_ids.append(row.DocumentID)
            # Convert the VARBINARY data back to a NumPy array
            embedding_np = np.frombuffer(row.Embedding, dtype=np.float32) 
            db_embeddings.append(embedding_np)

    if not db_embeddings:
        return {
            'max_similarity': 0.0,
            'match_count': 0,
            'risk_level': 'SIN REFERENCIAS',
            'risk_color': 'gray',
            'detailed_results': []
        }
    print(f"DEBUG: Dimensión del primer Embedding recuperado Modelo 3: XLM-RoBERTa: {embedding_np.shape}") 

    # 4. Calcular la similitud de TODAS las frases (new_embeddings) contra TODOS los documentos (db_embeddings)
    similarity_matrix = cosine_similarity(new_embeddings, db_embeddings)

    # 5. Encontrar la máxima similitud en toda la matriz
    max_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.0
    
    # 6. Determinar el nivel de riesgo
    risk_level, risk_color = determine_risk_level(max_similarity)
    
    # 7. Recopilar resultados detallados (el mejor match de fragmento para cada documento)
    detailed_results = {} 

    for doc_idx, doc_id in enumerate(db_ids):
        scores_for_doc = similarity_matrix[:, doc_idx]
        max_score_for_doc = np.max(scores_for_doc)
        sentence_idx = np.argmax(scores_for_doc)

        if max_score_for_doc >= PLAGIARISM_RISK_THRESHOLDS['BAJO']: # Umbral de 0.4
            
            detailed_results[doc_id] = {
                'document_id': doc_id,
                'similarity_score': max_score_for_doc,
                'matched_sentence': sentences[sentence_idx],
                'match_count': 1
            }
    print(f"DEBUG: Maxima Similitud Encontrada Modelo 3: XLM-RoBERTa: {max_similarity}")
    results_list = list(detailed_results.values())
    results_list.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        'max_similarity': max_similarity,
        'match_count': len(results_list),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'detailed_results': results_list
    }