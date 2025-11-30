# Detector de Plagio Semántico
## usa embeddings (vectores numéricos generados por modelos como sentence-transformers). 
# Utiliza la detección de plagio, especialmente el plagio semántico (por paráfrasis), 
# se basa en técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP)
## Estos vectores capturan el significado y el contexto.
## Al utilizar  el "modelo paraphrase-multilingual-mpnet-base-v2" 
# s tiene la capaDetección de Plagio Translingüe (Cross-Lingual Plagiarism Detection)

#Se recibe el contenido del documento como texto desde el controlador
# y se genera su embedding para compararlo con los embeddings almacenados en la base de datos SQL Server.

import gc
import torch
import pyodbc
import numpy as np
from sentence_transformers import SentenceTransformer
## Para comparar la similitud entre (embedding) entre el nuevo documento y los almacenados
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import CNXN_STR 

# --- GESTIÓN DE MEMORIA (LAZY LOADING) ---
# Variable global interna inicializada en None
_mpnet_model = None

def get_mpnet_model():
    """Carga el modelo solo si no existe en memoria."""
    global _mpnet_model
    if _mpnet_model is None:
        print("Cargando modelo MPNet (Global) en RAM... esto puede tardar un momento.")
        _mpnet_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
    return _mpnet_model

def unload_mpnet_model():
    """Libera la memoria RAM eliminando el modelo y forzando la recolección de basura."""
    global _mpnet_model
    if _mpnet_model is not None:
        print("Liberando memoria del modelo MPNet (Global)...")
        del _mpnet_model
        _mpnet_model = None
        gc.collect() # Fuerza a Python a liberar la RAM inmediatamente
        if torch.cuda.is_available(): # LIMPIEZA DE CACHÉ CUDA
            torch.cuda.empty_cache()

# Definición de la clasificación de riesgo basada en el umbrales de similitud
PLAGIARISM_RISK_THRESHOLDS = {
    'MUY ALTO': 0.85,
    'ALTO': 0.70,
    'MODERADO': 0.50,
    'BAJO': 0.40,
}

def determine_risk_level(score):
    """Clasifica el puntaje de similitud en un nivel de riesgo."""
    # En lugar de colores CSS, devolvemos nombres de clases CSS
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


def find_plagiarism_with_sql_server(document_content):
    try:
        # 1. Cargar el modelo bajo demanda
        model = get_mpnet_model()

        # Ahora usamos 'document_content' directamente para la codificación.
        new_embedding = model.encode(document_content)

        print(f"DEBUG: Dimensión del nuevo Embedding Modelo 1: MPNet (Global): {new_embedding.shape}") 

        # Connect to the database and get all embeddings
        with pyodbc.connect(CNXN_STR) as cnxn:
            cursor = cnxn.cursor()
            cursor.execute("SELECT DocumentID, Embedding FROM DocumentEmbeddings")
            
            # Load all document IDs and embeddings from the database
            db_embeddings = []
            db_ids = []
            for row in cursor.fetchall():
                db_ids.append(row.DocumentID)
                # Convert the VARBINARY data back to a NumPy array
                embedding_np = np.frombuffer(row.Embedding, dtype=np.float32) 
                if not db_embeddings: 
                     print(f"DEBUG: Dimensión del primer Embedding recuperado Modelo 1: MPNet (Global): {embedding_np.shape}")
                db_embeddings.append(embedding_np)

        # Check for empty database
        if not db_embeddings:
            # IMPORTANTE: Liberar memoria antes de salir
            unload_mpnet_model()
            return {
                'max_similarity': 0.0,
                'match_count': 0,
                'risk_level': 'SIN REFERENCIAS',
                'risk_color': 'gray',
                'detailed_results': []
            }

        # Calculate cosine similarity for the entire batch
        similarities = cosine_similarity([new_embedding], db_embeddings)[0]

        # Calcular la máxima similitud
        max_similarity = np.max(similarities) if similarities.size > 0 else 0.0
        print(f"DEBUG: Maxima Similitud Encontrada Modelo 1: MPNet (Global): {max_similarity}")

        # Determinar el nivel de riesgo
        risk_level, risk_color = determine_risk_level(max_similarity)
       
        results = []
        for i, score in enumerate(similarities):
            # Inicializamos siempre la variable para evitar errores
            matched_sentences = []

            # CORRECCIÓN: El if debe estar DENTRO del bucle for
            if score > 0.4:
                results.append({
                    'document_id': db_ids[i],
                    'similarity_score': score,
                    'matched_sentences': matched_sentences,
                    'match_count': len(matched_sentences)
                })

        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # 2. IMPORTANTE: Liberar memoria al finalizar el proceso exitoso
        unload_mpnet_model()

        return {
            'max_similarity': max_similarity,
            'match_count': len(results),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'detailed_results': results
        }

    except Exception as e:
        # En caso de error, aseguramos que la memoria se libere también
        print(f"Error en find_plagiarism_with_sql_server: {e}")
        unload_mpnet_model()
        # Re-lanzamos el error o devolvemos un objeto de error según prefieras
        raise e