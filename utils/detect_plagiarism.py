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

# --- SQL Server Connection ---
server = 'TALLER04\\TALLER04' 
database = 'plagioEmb' 
username = 'sa' 
password = 'HTObRrKy' 
driver = '{ODBC Driver 17 for SQL Server}' 

cnxn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# --- Load the embedding model ---
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')


# Renombramos new_document_path a document_content para mayor claridad
def find_plagiarism_with_sql_server(document_content):
  
    # Ahora usamos 'document_content' directamente para la codificación.
    new_embedding = model.encode(document_content)

    print(f"DEBUG: Dimensión del nuevo Embedding: {new_embedding.shape}") 

    # Connect to the database and get all embeddings
    with pyodbc.connect(cnxn_str) as cnxn:
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
        return []

    # Calculate cosine similarity for the entire batch
    similarities = cosine_similarity([new_embedding], db_embeddings)[0]


    # ------------------------------------------------------------------
    # LÍNEAS DE DEPURACIÓN A AÑADIR:
    max_similarity = np.max(similarities)
    print(f"DEBUG: Maxima Similitud Encontrada: {max_similarity}")
    # ---------------------

    results = []
    for i, score in enumerate(similarities):
        if score > 0.4:  # Threshold for plagiarism detection
            results.append({
                'document_id': db_ids[i],
                'similarity_score': score
            })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results