import io
from flask import render_template, request
from docx import Document
from utils.ai_detector import predict_ai_content, evaluate_model, LABELS

def get_ai_results():
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
            doc = Document(file_stream)
            for para in doc.paragraphs:
                document_content += para.text + "\n"
        except Exception as e:
            return render_template('ai_results.html', result=f"Error al procesar el archivo DOCX: {e}")
    else:
        return render_template('ai_results.html', result="Formato de archivo no soportado.")

    prediction = predict_ai_content(document_content)
    evaluation_report_string, accuracy_value = evaluate_model(LABELS) 

    return render_template('ai_results.html',
                           result=prediction,
                           report=evaluation_report_string,
                           accuracy=accuracy_value)