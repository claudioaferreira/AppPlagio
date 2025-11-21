# controllers/plagiarism_controller.py

import io
import sys
from flask import render_template, request
from docx import Document

# Detectores
from utils.plagioDetector.detect_plagiarism import (
    find_plagiarism_with_sql_server as mpnet_full_doc_detector
)
from utils.plagioDetector.detect_plagiarism_granular import (
    find_plagiarism_with_roberta_segmentation as mpnet_granular_detector
)
from utils.plagioDetector.detect_plagiarismRobertaEmbeddings import (
    find_plagiarism_with_roberta_embeddings as roberta_granular_detector
)


def build_empty_results(status="SIN DATOS"):
    return {
        "max_similarity": 0.0,
        "risk_level": status,
        "risk_color": "gray",
        "detailed_results": [],
        "match_count": 0
    }



def normalize_model_output(raw):
    """
    Garantiza que cada modelo produzca la estructura esperada por la plantilla.
    """
    if not isinstance(raw, dict):
        return {
            "max_similarity": 0.0,
            "risk_level": "SIN DATOS",
            "risk_color": "gray",
            "detailed_results": [],
            "match_count": 0
        }

    detailed = [
        {
            "document_id": item.get("document_id"),
            "similarity_score": item.get("similarity_score", 0.0),
            "matched_sentence": item.get("matched_sentence", "")
        }
        for item in raw.get("detailed_results", [])
        if isinstance(item, dict)
    ]

    return {
        "max_similarity": raw.get("max_similarity", 0.0),
        "risk_level": raw.get("risk_level", "SIN DATOS"),
        "risk_color": raw.get("risk_color", "gray"),
        "detailed_results": detailed,
        "match_count": raw.get("match_count", len(detailed))
    }


def get_plagiarism_results():
    # -----------------------------
    # VALIDACIÓN DEL ARCHIVO
    # -----------------------------
    if "documento" not in request.files or request.files["documento"].filename == "":
        empty = build_empty_results("SIN DOCUMENTO")
        return render_template(
            "detectorPlagio/plagio_results.html",
            model1_results=empty,
            model2_results=empty,
            model3_results=empty,
            analysis_mode="none"
        )

    file = request.files["documento"]
    file_stream = io.BytesIO(file.read())
    file_ext = file.filename.rsplit(".", 1)[1].lower()

    document_content = ""

    # -----------------------------
    # PROCESADO TXT / DOCX
    # -----------------------------
    if file_ext == "txt":
        document_content = file_stream.getvalue().decode("utf-8")

    elif file_ext == "docx":
        try:
            doc = Document(file_stream)
            document_content = "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            print(f"Error procesando DOCX: {e}", file=sys.stderr)
            err = build_empty_results("ERROR ARCHIVO")
            return render_template(
                "detectorPlagio/plagio_results.html",
                model1_results=err,
                model2_results=err,
                model3_results=err,
                analysis_mode="none"
            )

    else:
        unsupported = build_empty_results("FORMATO NO SOPORTADO")
        return render_template(
            "detectorPlagio/plagio_results.html",
            model1_results=unsupported,
            model2_results=unsupported,
            model3_results=unsupported,
            analysis_mode="none"
        )

    # -----------------------------
    # MODO SELECCIONADO EN EL FORM
    # -----------------------------
    analysis_mode = request.form.get("analysis_mode", "all").strip()

    # Estructuras iniciales
    model1_results = build_empty_results()
    model2_results = build_empty_results()
    model3_results = build_empty_results()

    # -----------------------------
    # EJECUCIÓN DE MODELOS
    # -----------------------------
    if analysis_mode in ["all", "global"]:
        raw = mpnet_full_doc_detector(document_content)
        model1_results = normalize_model_output(raw)

    if analysis_mode in ["all", "granular_mpnet"]:
        raw = mpnet_granular_detector(document_content)
        model2_results = normalize_model_output(raw)

    if analysis_mode in ["all", "granular_roberta"]:
        raw = roberta_granular_detector(document_content)
        model3_results = normalize_model_output(raw)

    # -----------------------------
    # RENDER FINAL
    # -----------------------------
    return render_template(
        "detectorPlagio/plagio_results.html",
        model1_results=model1_results,
        model2_results=model2_results,
        model3_results=model3_results,
        analysis_mode=analysis_mode
    )
