# Detector de Plagio Semántico
## usa embeddings (vectores numéricos generados por modelos como sentence-transformers). 
# Utiliza la detección de plagio, especialmente el plagio semántico (por paráfrasis), 
# se basa en técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP)
## Estos vectores capturan el significado y el contexto.
## Al utilizar  el "modelo paraphrase-multilingual-mpnet-base-v2" 
# s tiene la capaDetección de Plagio Translingüe (Cross-Lingual Plagiarism Detection)

#Se recibe el contenido del documento como texto desde el controlador
# y se genera su embedding para compararlo con los embeddings almacenados en la base de datos SQL Server.


import pyodbc
import numpy as np
from sentence_transformers import SentenceTransformer
##Para comparar la similitud entre (embedding) entre el nuevo documento y los almacenados
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import CNXN_STR 
# Cargar el modelo de SentenceTransformer una vez al inicio
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

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
        return "MUY ALTO", "risk-muy-alto" # <-- Clase CSS
    elif score >= PLAGIARISM_RISK_THRESHOLDS['ALTO']:
        return "ALTO", "risk-alto"         # <-- Clase CSS
    elif score >= PLAGIARISM_RISK_THRESHOLDS['MODERADO']:
        return "MODERADO", "risk-moderado" # <-- Clase CSS
    elif score >= PLAGIARISM_RISK_THRESHOLDS['BAJO']:
        return "BAJO", "risk-bajo"         # <-- Clase CSS
    else:
        return "INSIGNIFICANTE", "risk-insignificante" # <-- Clase CSS


# Renombramos new_document_path a document_content para mayor claridad
def find_plagiarism_with_sql_server(document_content):
  
    # Ahora usamos 'document_content' directamente para la codificación.
    new_embedding = model.encode(document_content)

    print(f"DEBUG: Dimensión del nuevo Embedding: {new_embedding.shape}") 

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
                 print(f"DEBUG: Dimensión del primer Embedding recuperado: {embedding_np.shape}")
            db_embeddings.append(embedding_np)

    # Check for empty database
    if not db_embeddings:
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
    print(f"DEBUG: Maxima Similitud Encontrada: {max_similarity}")

    # Determinar el nivel de riesgo
    risk_level, risk_color = determine_risk_level(max_similarity)
   

    results = []
    for i, score in enumerate(similarities):
        if score > 0.4:  # Threshold for plagiarism detection
            results.append({
                'document_id': db_ids[i],
                'similarity_score': score
            })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return {
        'max_similarity': max_similarity,
        'match_count': len(results),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'detailed_results': results
    }