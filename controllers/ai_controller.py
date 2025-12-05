import io
from flask import render_template, request
from docx import Document
from pypdf import PdfReader

# ================================
#  MODELO 1 — LightGBM TF-IDF
# ================================
from utils.detectorIA.ai_detectorLGBMClassifierFIX import (
  predict_ai_content as predict_tfidf_lgbm,
  evaluate_model as evaluate_tfidf_lgbm,
  get_text_stats as get_stats_tfidf_lgbm,
  LABELS as LABELS_TFIDF
)

# ================================================
#  MODELO 2 — LightGBM + Feature Engineering (FE)
# ================================================
from utils.detectorIA.ai_detectorLGBMClassifierFE import (
  predict_ai_content as predict_lgbm_fe,
  evaluate_model as evaluate_lgbm_fe,
  LABELS as LABELS_FE
)

def get_ai_results():

  # Validación del archivo
  if 'documento' not in request.files or request.files['documento'].filename == '':
    return render_template('detectorContenidoIA/ai_results.html', result="No se seleccionó ningún archivo.")

  file = request.files['documento']
  file_stream = io.BytesIO(file.read())
  file_ext = file.filename.rsplit('.', 1)[1].lower()

  document_content = ""

  # -------------------------
  #  LECTURA DEL CONTENIDO
  # -------------------------
  if file_ext == 'txt':
    document_content = file_stream.getvalue().decode('utf-8')

  elif file_ext == 'docx':
    try:
      doc = Document(file_stream)
      for para in doc.paragraphs:
        document_content += para.text + "\n"
    except Exception as e:
      return render_template('detectorContenidoIA/ai_results.html', result=f"Error al procesar el archivo DOCX: {e}")

  elif file_ext == 'pdf':
    try:
      reader = PdfReader(file_stream)
      for page in reader.pages:
        text = page.extract_text()
        if text:
          document_content += text + "\n"
    except Exception as e:
      return render_template('detectorContenidoIA/ai_results.html', result=f"Error al procesar el archivo PDF: {e}")

  else:
    return render_template('detectorContenidoIA/ai_results.html', result="Formato de archivo no soportado.")

  # ====================================================================
  #         EJECUCIÓN DE LOS MODELOS DE IA
  # ====================================================================

  # ---------------------------
  #  1. LGBM TF-IDF (modelo base)
  # ---------------------------
  # prediction_tfidf es un diccionario: {"label": "...", "ai_score": ...}
  prediction_tfidf = predict_tfidf_lgbm(document_content) 
  report_tfidf, accuracy_tfidf = evaluate_tfidf_lgbm(LABELS_TFIDF)

  stats = get_stats_tfidf_lgbm(document_content) 

  # -----------------------------------------------
  #  2. LGBM + Feature Engineering (modelo pro)
  # -----------------------------------------------
  # prediction_fe ahora es un diccionario: {"label": "...", "ai_score": ...}
  prediction_fe = predict_lgbm_fe(document_content) 
  report_fe, accuracy_fe = evaluate_lgbm_fe(LABELS_FE)

  # -----------------------
  #  RENDERIZAR RESULTADO
  # -----------------------
  return render_template('detectorContenidoIA/ai_results.html', 
             # === RESULTADO PRINCIPAL (Gauge y Veredicto) ===
             # La plantilla usa 'result_data' para el gauge/veredicto principal
             result_data=prediction_fe,       
             
             # === DATOS PARA LA COMPARATIVA Y ESTADÍSTICAS ===
             # Modelo Estándar (panel izquierdo)
             result_tfidf_lgbm=prediction_tfidf, 
             
             # Modelo Avanzado (panel derecho)
             result_lgbm_fe=prediction_fe,     
             
             # Estadísticas del texto
             stats=stats,              
             # Texto original
             original_text=document_content 
             )