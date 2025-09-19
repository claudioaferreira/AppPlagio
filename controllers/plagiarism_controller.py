# plagiarism_controller.py

import io
import sys
from flask import render_template, request
from docx import Document
from utils.detect_plagiarism import find_plagiarism_with_sql_server

def get_plagiarism_results():
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

    # Call the logic from the detect_plagiarism.py file
    plagiarism_results = find_plagiarism_with_sql_server(document_content)

    return render_template('results.html',
                           results=plagiarism_results,
                           highest_similarity=plagiarism_results[0]['similarity_score'] if plagiarism_results else 0)