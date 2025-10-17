import io
from flask import render_template, request
from docx import Document
#funciones de Regresión Logística
from utils.ai_detector import predict_ai_content as predict_lr, evaluate_model as evaluate_lr, LABELS
#funciones de LightGBM
from utils.ai_detectorLGBMClassifierFIX import predict_ai_content as predict_lgbm, evaluate_model as evaluate_lgbm 
#funciones de LightGBM+Feature Engineering
from utils.ai_detectorLGBMClassifierFE import StylometricFeatureExtractor, predict_ai_content as predict_lgbm_fe, evaluate_model as evaluate_lgbm_fe


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


      # 1. Predicción y Evaluación de Regresión Logística (LR)
    prediction_lr = predict_lr(document_content)
    report_lr, accuracy_lr = evaluate_lr(LABELS) 

    # 2. Predicción y Evaluación de LightGBM (LGBM)
    prediction_lgbm = predict_lgbm(document_content)
    report_lgbm, accuracy_lgbm = evaluate_lgbm(LABELS)

    # 3. Predicción y Evaluación de LightGBM + Feature Engineering (LGBM_FE)
    result_lgbm_fe = predict_lgbm_fe(document_content)
    report_lgbm_fe, accuracy_lgbm_fe = evaluate_lgbm_fe(LABELS)

    return render_template('ai_results.html',
                           # Datos de Regresión Logística
                           result_lr=prediction_lr,
                           report_lr=report_lr,
                           accuracy_lr=accuracy_lr,
                           # Datos de LightGBM
                           result_lgbm=prediction_lgbm,
                           report_lgbm=report_lgbm,
                           accuracy_lgbm=accuracy_lgbm,
                           # Datos de LightGBM + Feature Engineering
                           result_lgbm_fe=result_lgbm_fe,
                           report_lgbm_fe=report_lgbm_fe,
                           accuracy_lgbm_fe=accuracy_lgbm_fe
                           )