import os
import io
import sys
import docx
from flask import Flask, render_template, request
import pyodbc
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
from ai_detector import predict_ai_content, evaluate_model, LABELS
from paraphraser import paraphrase_text

# --- Conexión a SQL Server ---
server = 'TALLER04\\TALLER04' 
database = 'plagioEmb' 
username = 'sa' 
password = 'HTObRrKy' 
driver = '{ODBC Driver 17 for SQL Server}'
cnxn_str = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

app = Flask(__name__)

# --- Carga el modelo de embedding ---
try:
    model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
except Exception as e:
    print(f"Error al cargar el modelo de embedding: {e}", file=sys.stderr)
    sys.exit(1)
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'documento' not in request.files or request.files['documento'].filename == '':
        return render_template('results.html', highest_similarity=0, message="No se seleccionó ningún archivo.")

    file = request.files['documento']
    file_stream = io.BytesIO(file.read())
    file_ext = file.filename.rsplit('.', 1)[1].lower()

    if file_ext == 'txt':
        document_content = file_stream.getvalue().decode('utf-8')
    elif file_ext == 'docx':
        try:
            doc = Document(file_stream)
            document_content = ""
            for para in doc.paragraphs:
                document_content += para.text + "\n"
        except Exception as e:
            print(f"Error procesando el archivo DOCX: {e}", file=sys.stderr)
            return render_template('results.html', highest_similarity=0, message="Error al procesar el archivo DOCX. Por favor, asegúrese de que es un archivo válido.")
    else:
        return render_template('results.html', highest_similarity=0, message="Formato de archivo no soportado. Por favor, suba un archivo .txt o .docx.")

    plagiarism_results = find_plagiarism(document_content)

    return render_template('results.html', 
                           results=plagiarism_results, 
                           highest_similarity=plagiarism_results[0]['similarity_score'] if plagiarism_results else 0)


def find_plagiarism(document_content):
    if not document_content.strip():
        return []

    # Genera el embedding del nuevo documento
    new_embedding = model.encode(document_content)

    # Conecta a la base de datos para obtener todos los embeddings
    try:
        with pyodbc.connect(cnxn_str) as cnxn:
            cursor = cnxn.cursor()
            cursor.execute("SELECT DocumentID, Embedding FROM DocumentEmbeddings")
            
            db_embeddings = []
            db_ids = []
            for row in cursor.fetchall():
                db_ids.append(row.DocumentID)
                embedding_np = np.frombuffer(row.Embedding, dtype=np.float32)
                db_embeddings.append(embedding_np)
    except pyodbc.Error as ex:
        sqlstate = ex.args[0]
        print(f"Error de conexión o de base de datos: {sqlstate}", file=sys.stderr)
        return []

    if not db_embeddings:
        return []

    # Calcula la similitud del coseno para todo el lote
    similarities = cosine_similarity([new_embedding], db_embeddings)[0]

    results = []
    for i, score in enumerate(similarities):
        if score > 0.3:  # Umbral de similitud
            results.append({
                'document_id': db_ids[i],
                'similarity_score': score * 100
            })

    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Devuelve solo los 5 resultados con mayor similitud
    return results[:5]



# RUTA PARAFRASEADOR
@app.route('/paraphraser-page')
def paraphraser_page():
    """Ruta para la página del parafraseador."""
    return render_template('paraphraser.html')

@app.route('/paraphrase', methods=['POST'])
def paraphrase():
    """Ruta para procesar el texto y parafrasearlo."""
    original_text = request.form['original_text']
    
    # Llama a la función de tu nuevo script
    paraphrased_text = paraphrase_text(original_text)
    
    return render_template('paraphrased_result.html', original_text=original_text, 
                           paraphrased_text=paraphrased_text)



#DETECTOR IA

@app.route('/ai-detector-page')
def ai_detector_page():
    """Ruta para la página del detector de IA."""
    return render_template('ai_detector.html')

@app.route('/ai-detect', methods=['POST'])
def ai_detect():
    """Ruta para procesar el documento y detectar contenido de IA."""
    if 'documento' not in request.files or request.files['documento'].filename == '':
        return render_template('ai_results.html', result="No se seleccionó ningún archivo.")

    file = request.files['documento']
    file_stream = io.BytesIO(file.read())
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    
    document_content = ""
    if file_ext == 'txt':
        document_content = file_stream.getvalue().decode('utf-8')
    elif file_ext == 'docx':
        try:
            doc = docx.Document(file_stream)
            for para in doc.paragraphs:
                document_content += para.text + "\n"
        except Exception as e:
            return render_template('ai_results.html', result=f"Error al procesar el archivo DOCX: {e}")
    else:
        return render_template('ai_results.html', result="Formato de archivo no soportado.")

    # Llama a la función de tu nuevo script
    prediction = predict_ai_content(document_content)
    evaluation_report = evaluate_model(LABELS)
    
    return render_template('ai_results.html', 
                           result=prediction,
                           report=evaluation_report)

    

if __name__ == '__main__':
    # Flask ya crea el directorio 'static' y busca plantillas en 'templates'.
    # La carpeta 'uploads' no es necesaria con el manejo de streams.
    app.run(debug=True)


    