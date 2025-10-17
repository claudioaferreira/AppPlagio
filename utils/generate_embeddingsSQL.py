import os
import json
import pyodbc
from sentence_transformers import SentenceTransformer

# Carga el modelo de embedding multilingüe (se descarga automáticamente la primera vez)
print("Cargando el modelo de embedding...")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# Ruta a la carpeta del corpus 2011
corpus_path = os.path.join(os.getcwd(), 'pan-plagiarism-corpus-2011', 'external-detection-corpus', 'source-document')
# Ruta a la carpeta del corpus 2025
corpus_path_2025 = os.path.join(os.getcwd(), 'pan-plagiarism-corpus-2025', 'external-detection-corpus', 'source-document')
corpus_path_new_pan25_src = os.path.join(os.getcwd(), 'pan25-generated-plagiarism-detection-train', '01_train', '01_train', 'src')


# --- CONFIGURACIÓN DE LA BASE DE DATOS SQL SERVER ---
# Reemplaza estos valores con la información de tu servidor
server = 'TALLER04\\TALLER04' 
database = 'plagioEmb' 
username = 'sa' 
password = 'HTObRrKy' 
driver = '{ODBC Driver 17 for SQL Server}' # Asegúrate de que este es el driver correcto

# Cadena de conexión
cnxn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

# --- CONEXIÓN Y PROCESAMIENTO ---
try:
    print("Conectando a la base de datos SQL Server...")
    cnxn = pyodbc.connect(cnxn_str)
    cursor = cnxn.cursor()
    
    print("Conexión exitosa. Recorriendo documentos...")
    
    # Recorre los documentos del corpus
    for root, dirs, files in os.walk(corpus_path_new_pan25_src):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_content = f.read()

                # Genera el embedding
                embedding = model.encode(document_content)
                
                # Prepara los datos para la inserción
                document_id = file
                embedding_bytes = embedding.tobytes() # Convierte el array de numpy a bytes para VARBINARY
                
                # Inserta el documento y su embedding en la tabla
                insert_query = "INSERT INTO DocumentEmbeddings (DocumentID, DocumentText, Embedding) VALUES (?, ?, ?)"
                cursor.execute(insert_query, (document_id, document_content, embedding_bytes))
                cnxn.commit()
                print(f"Documento '{document_id}' insertado.")

    print("\nProceso completado. Todos los embeddings han sido guardados en la base de datos.")

except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Error de conexión o de base de datos: {sqlstate}")
    print(ex)

finally:
    if 'cnxn' in locals():
        cnxn.close()