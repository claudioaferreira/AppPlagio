import os
import json
from sentence_transformers import SentenceTransformer

# Carga el modelo de embedding multilingüe (se descarga automáticamente la primera vez)
print("Cargando el modelo de embedding...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Ruta a la carpeta del corpus (asegúrate de que sea la ruta correcta en tu sistema)
corpus_path = os.path.join(os.getcwd(), 'pan-plagiarism-corpus-2011', 'external-detection-corpus', 'source-document')

# Diccionario para guardar los embeddings
embeddings_db = {}

# Recorre los documentos del corpus
print("Procesando documentos y generando embeddings...")
for root, dirs, files in os.walk(corpus_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                document_content = f.read()
            
            # Genera el embedding
            embedding = model.encode(document_content)
            
            # Guarda el embedding en el diccionario con el nombre del archivo como clave
            embeddings_db[file] = embedding.tolist() # Convierte a lista para guardarlo en JSON
            print(f"Embedding generado para: {file}")

# Guarda la base de datos de embeddings en un archivo local
output_file = 'local_embeddings.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(embeddings_db, f, ensure_ascii=False, indent=4)

print(f"\nProceso completado. Embeddings guardados en '{output_file}'")