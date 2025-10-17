import os
import sys 
import pyodbc
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# --- INICIO DE CORRECCIÓN DE RUTA DE IMPORTACIÓN ---
# Ajusta el path para permitir la importación de módulos en la raíz del proyecto (e.g., database)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..')
if project_root not in sys.path:
    sys.path.append(project_root)
# ----------------------------------------------------

# Importa la cadena de conexión definida en database/connection.py
from database.connection import CNXN_STR 

# --- CONFIGURACIÓN DE MODELO Y TABLA ---
MODEL_NAME = 'stsb-xlm-r-multilingual'
TABLE_NAME = 'DocumentEmbeddings_XLM_R' # Nueva tabla para XLM-RoBERTa

# --- CONFIGURACIÓN DE RUTAS DEL CORPUS ---
# Define la ruta base para ambos corpus usando os.getcwd() (donde ejecutas python app.py)
base_path = os.getcwd() 
corpus_path_2011 = os.path.join(base_path, 'pan-plagiarism-corpus-2011', 'external-detection-corpus', 'source-document')
corpus_path_2025 = os.path.join(base_path, 'pan-plagiarism-corpus-2025', 'external-detection-corpus', 'source-document')
corpus_path_new_pan25_src = os.path.join(base_path, 'pan25-generated-plagiarism-detection-train', '01_train', '01_train', 'src')


# Lista de directorios a recorrer
corpus_directories = [corpus_path_new_pan25_src]


# --- FUNCIÓN DE INSERCIÓN DEDICADA ---
def insert_embedding(cnxn, doc_id, doc_text, embedding):
    """Inserta el embedding en la tabla DocumentEmbeddings_XLM_R."""
    
    # Convierte el array NumPy a bytes (float32) para VARBINARY en SQL Server
    embedding_bytes = embedding.astype(np.float32).tobytes()

    # Sentencia SQL para la inserción
    sql_insert = f"""
    INSERT INTO {TABLE_NAME} (DocumentID, DocumentText, Embedding)
    VALUES (?, ?, ?)
    """
    
    try:
        cursor = cnxn.cursor()
        cursor.execute(sql_insert, doc_id, doc_text, embedding_bytes)
        cnxn.commit()
    except pyodbc.IntegrityError as e:
        # Maneja la advertencia si ya existe un DocumentID (clave primaria)
        print(f"ADVERTENCIA: Documento '{doc_id}' ya existe en {TABLE_NAME}. Saltando. Error: {e}", file=sys.stderr)
        cnxn.rollback()
    except Exception as e:
        print(f"ERROR al insertar en {TABLE_NAME} para {doc_id}: {e}", file=sys.stderr)
        cnxn.rollback()


def generate_and_store_xlmr_embeddings():
    """Carga los documentos desde archivos TXT usando os.walk, genera los embeddings y los guarda."""
    
    print(f"Iniciando la generación de embeddings para el modelo: {MODEL_NAME}")
    
    try:
        model = SentenceTransformer(MODEL_NAME)
        print("Modelo XLM-RoBERTa cargado con éxito.")
    except Exception as e:
        print(f"ERROR al cargar el modelo SentenceTransformer: {e}", file=sys.stderr)
        return

    # Conectar a la base de datos
    try:
        cnxn = pyodbc.connect(CNXN_STR)
        print("Conexión a la base de datos establecida.")
    except Exception as e:
        print(f"ERROR al conectar a la base de datos: {e}", file=sys.stderr)
        return

    # Recorrer los directorios usando la lógica de os.walk del usuario
    for corpus_dir in corpus_directories:
        if not os.path.isdir(corpus_dir):
            print(f"ADVERTENCIA: El directorio del corpus no existe: {corpus_dir}. Saltando.", file=sys.stderr)
            continue
            
        # Recorre la estructura de carpetas (part1, part2, etc.)
        for root, dirs, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            document_content = f.read()
                    except Exception as e:
                        print(f"ERROR al leer el archivo {file_path}: {e}", file=sys.stderr)
                        continue

                    # El DocumentID es el nombre del archivo (e.g., 'source-document00001.txt')
                    document_id = file 
                    
                    if document_content and len(document_content.strip()) > 0:
                        print(f"Procesando documento ID: {document_id}")
                        
                        # 1. Generar el embedding XLM-RoBERTa
                        embedding = model.encode(document_content)

                        # 2. Insertar en la tabla DocumentEmbeddings_XLM_R
                        insert_embedding(cnxn, document_id, document_content, embedding)
                    else:
                        print(f"ADVERTENCIA: Saltando {document_id} por archivo vacío.")

    cnxn.close()
    print("\nProceso de generación y almacenamiento de embeddings de XLM-RoBERTa completado con éxito.")


if __name__ == '__main__':
    generate_and_store_xlmr_embeddings()