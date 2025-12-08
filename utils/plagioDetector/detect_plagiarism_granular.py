#Modelo 2: MPNet (Fragmentado) -Este nuevo detector implementa la lógica de segmentación por frases () (sent_tokenize) 
# y compara cada frase del documento nuevo contra todos los embeddings de los documentos en la base de datos SQL.


import gc
import pyodbc
import torch
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import CNXN_STR

# --- GESTIÓN DE MEMORIA ---
_mpnet_granular_model = None

def get_mpnet_granular_model():
    """Carga el modelo MPNet solo si no existe en memoria."""
    global _mpnet_granular_model
    if _mpnet_granular_model is None:
        print("Cargando modelo MPNet (Granular)...")
        # CORRECCIÓN 1: Forzamos CPU para evitar caídas por Timeout de GPU
        _mpnet_granular_model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    return _mpnet_granular_model

def unload_mpnet_granular_model():
    """Libera la memoria RAM."""
    global _mpnet_granular_model
    try:
        if _mpnet_granular_model is not None:
            print("Liberando memoria del modelo MPNet (Granular)...")
            del _mpnet_granular_model
            _mpnet_granular_model = None
        
        gc.collect()
        
        # CORRECCIÓN 2: Protección contra error de CUDA al limpiar
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass # Ignorar error si la GPU ya murió
    except Exception as e:
        print(f"Nota: Limpieza parcial ({e})")

# Umbrales
PLAGIARISM_RISK_THRESHOLDS = {
    'MUY ALTO': 0.85,
    'ALTO': 0.70,
    'MODERADO': 0.50,
    'BAJO': 0.40,
}

def determine_risk_level(score):
    if score >= PLAGIARISM_RISK_THRESHOLDS['MUY ALTO']: return "MUY ALTO", "risk-muy-alto"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['ALTO']: return "ALTO", "risk-alto"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['MODERADO']: return "MODERADO", "risk-moderado"
    elif score >= PLAGIARISM_RISK_THRESHOLDS['BAJO']: return "BAJO", "risk-bajo"
    else: return "INSIGNIFICANTE", "risk-insignificante"


def find_plagiarism_with_roberta_segmentation(document_content):
    """
    Detecta plagio granular usando MPNet.
    """
    try:
        # 1. Segmentación
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

        # 2. Cargar modelo (CPU MODE)
        model = get_mpnet_granular_model()

        # Codificar (sin GPU crash)
        new_embeddings = model.encode(sentences, show_progress_bar=True)
        
        # 3. Base de datos
        with pyodbc.connect(CNXN_STR) as cnxn:
            cursor = cnxn.cursor()
            
            cursor.execute("SELECT DocumentID, Embedding FROM DocumentEmbeddings")
            
            db_embeddings = []
            db_ids = []

            for row in cursor.fetchall():
                db_ids.append(row.DocumentID)
                embedding_np = np.frombuffer(row.Embedding, dtype=np.float32)
                db_embeddings.append(embedding_np)

        if not db_embeddings:
            unload_mpnet_granular_model()
            return {
                'max_similarity': 0.0,
                'match_count': 0,
                'risk_level': 'SIN REFERENCIAS',
                'risk_color': 'gray',
                'detailed_results': []
            }

        # 4. Similitud
        similarity_matrix = cosine_similarity(new_embeddings, db_embeddings)

        # 5. Resultados
        max_similarity = np.max(similarity_matrix) if similarity_matrix.size > 0 else 0.0
        risk_level, risk_color = determine_risk_level(max_similarity)
        
        detailed_results = {}
        for doc_idx, doc_id in enumerate(db_ids):
            scores_for_doc = similarity_matrix[:, doc_idx]
            max_score_for_doc = np.max(scores_for_doc)
            sentence_idx = np.argmax(scores_for_doc)

            if max_score_for_doc >= PLAGIARISM_RISK_THRESHOLDS['BAJO']:
                detailed_results[doc_id] = {
                    'document_id': doc_id,
                    'similarity_score': max_score_for_doc,
                    'matched_sentence': sentences[sentence_idx],
                    'match_count': 1
                }

        results_list = list(detailed_results.values())
        results_list.sort(key=lambda x: x['similarity_score'], reverse=True)

        unload_mpnet_granular_model()

        return {
            'max_similarity': max_similarity,
            'match_count': len(results_list),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'detailed_results': results_list
        }

    except Exception as e:
        print(f"Error en granular detector: {e}")
        unload_mpnet_granular_model()
        raise e