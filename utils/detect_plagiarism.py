# Detector de Plagio Semántico
## usa embeddings (vectores numéricos generados por modelos como sentence-transformers). 
## Estos vectores capturan el significado y el contexto.

import pyodbc
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from database.connection import CNXN_STR 



# --- Load the embedding model ---
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def find_plagiarism_with_sql_server(new_document_path):
    # Read and encode the new document
    with open(new_document_path, 'r', encoding='utf-8') as f:
        new_document_content = f.read()
    new_embedding = model.encode(new_document_content)

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
            db_embeddings.append(embedding_np)

    # Check for empty database
    if not db_embeddings:
        return []

    # Calculate cosine similarity for the entire batch
    similarities = cosine_similarity([new_embedding], db_embeddings)[0]

    results = []
    for i, score in enumerate(similarities):
        if score > 0.7:  # Threshold for plagiarism detection
            results.append({
                'document_id': db_ids[i],
                'similarity_score': score
            })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results

# Example usage
# plagiarism_results = find_plagiarism_with_sql_server('my_document.txt')
# ... print results