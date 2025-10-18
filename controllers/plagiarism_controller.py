# controllers/plagiarism_controller.py

import io
import sys
from flask import render_template, request
from docx import Document
# Importamos los tres detectores con alias descriptivos para el código
from utils.detect_plagiarism import find_plagiarism_with_sql_server as mpnet_full_doc_detector
from utils.detect_plagiarism_granular import find_plagiarism_with_roberta_segmentation as mpnet_granular_detector
from utils.detect_plagiarismRobertaEmbeddings import find_plagiarism_with_roberta_embeddings as roberta_granular_detector

def get_plagiarism_results():
    # Estructura de resultados vacíos/error para evitar fallos en el template
    empty_results = {'max_similarity': 0, 'risk_level': "SIN DATOS", 'risk_color': "gray", 'detailed_results': []}

    # --- Manejo de archivos y errores ---
    if 'documento' not in request.files or request.files['documento'].filename == '':
        empty_results['risk_level'] = "SIN DOCUMENTO"
        # Devolvemos los tres modelos con resultados vacíos
        return render_template('results.html', model1_results=empty_results, model2_results=empty_results, model3_results=empty_results)

    file = request.files['documento']
    file_stream = io.BytesIO(file.read())
    file_ext = file.filename.rsplit('.', 1)[1].lower()
    
    document_content = ""
    
    if file_ext == 'txt':
        document_content = file_stream.getvalue().decode('utf-8')
    elif file_ext == 'docx':
        try:
            doc = Document(file_stream)
            for para in doc.paragraphs:
                document_content += para.text + "\n"
        except Exception as e:
            print(f"Error procesando el archivo DOCX: {e}", file=sys.stderr)
            error_results = empty_results.copy()
            error_results['risk_level'] = "ERROR ARCHIVO"
            return render_template('results.html', model1_results=error_results, model2_results=error_results, model3_results=error_results)
    else:
        unsupported_results = empty_results.copy()
        unsupported_results['risk_level'] = "FORMATO NO SOPORTADO"
        return render_template('results.html', model1_results=unsupported_results, model2_results=unsupported_results, model3_results=unsupported_results)
    # ------------------------------------

    # 1. Modelo 1: MPNet - Documento completo (Detector original)
    model1_results = mpnet_full_doc_detector(document_content)

    # 2. Modelo 2: MPNet - Segmentación por frases (Granular)
    model2_results = mpnet_granular_detector(document_content)
    
    # 3. Modelo 3: XLM-RoBERTa - Segmentación por frases (Nuevo y avanzado)
    model3_results = roberta_granular_detector(document_content)

    # 4. Renderizar la plantilla con los TRES resultados
    return render_template('detectorPlagio/plagio_results.html', 
        model1_results=model1_results,
        model2_results=model2_results,
        model3_results=model3_results)