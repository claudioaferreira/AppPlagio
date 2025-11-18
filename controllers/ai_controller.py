import io
from flask import render_template, request
from docx import Document
#funciones de Regresión Logística
  #from utils.ai_detector import predict_ai_content as predict_lr, evaluate_model as evaluate_lr, LABELS
#funciones de LightGBM
from utils.detectorIA.ai_detectorLGBMClassifierFIX import predict_ai_content as predict_main, get_text_stats, evaluate_model as evaluate_main, LABELS
#funciones de LightGBM+Feature Engineering
  #from utils.ai_detectorLGBMClassifierFE import StylometricFeatureExtractor, predict_ai_content as predict_lgbm_fe, evaluate_model as evaluate_lgbm_fe
from utils.detectorIA.ai_detectorRoBERTa import predict_ai_content as predict_roberta, get_text_stats

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
    #prediction_lr = predict_lr(document_content)
    #report_lr, accuracy_lr = evaluate_lr(LABELS) 

    # 2. Predicción y Evaluación de LightGBM (LGBM)
    #prediction_lgbm = predict_lgbm(document_content)
    #report_lgbm, accuracy_lgbm = evaluate_lgbm(LABELS)

    # 3. Predicción y Evaluación de LightGBM + Feature Engineering (LGBM_FE)
    #result_lgbm_fe = predict_lgbm_fe(document_content)
    #report_lgbm_fe, accuracy_lgbm_fe = evaluate_lgbm_fe(LABELS)
    result_data = predict_main(document_content)
    text_stats = get_text_stats(document_content)
    report_string, accuracy = evaluate_main(LABELS)

    # 4. Predicción con RoBERTa
    result_roberta = predict_roberta(document_content)
    text_stats_roberta = get_text_stats(document_content)

    return render_template('detectorContenidoIA/ai_results.html',
                           original_text=document_content,      # El texto original para mostrarlo
                           result_data=result_data,           # El diccionario con el puntaje y la etiqueta
                           stats=text_stats,                  # El diccionario con palabras/caracteres
                           
                           # (Opcional) Si aún quieres mostrar el reporte técnico
                           report_lgbm_fe=report_string,
                           accuracy_lgbm_fe=accuracy

                           #RoBERTa
                            ,result_roberta=result_roberta
                            ,stats_roberta=text_stats_roberta
                          )