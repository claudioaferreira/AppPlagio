# controllers/math_solve_controller.py

from flask import render_template, request
import google.generativeai as genai
from pydantic import BaseModel, Field   # Para JSON structuring
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Configurar la API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ==========================================================
# 1. ESQUEMA PYDANTIC PARA JSON STRUCTURING
# ==========================================================
class MathSolution(BaseModel):
    problem: str = Field(..., description="Problem statement sent by the user")
    solution: str = Field(..., description="Step-by-step solution in text")
    result: str = Field(..., description="Final numeric or symbolic result")

# Crear una instancia del modelo
model = genai.GenerativeModel(
    "gemini-1.5-pro",
    generation_config={
        "temperature": 0.4,
        "max_output_tokens": 800,
    }
)

# ==========================================================
# 2. FUNCIÓN PRINCIPAL
# ==========================================================
def test_ai_functionality():
    ai_data = None
    error_message = None
    user_content = None

    if request.method == "POST":
        user_content = request.form.get("content", "")

        try:
            # Guardr esquema para JSON Structuring
            structured_model = model.with_structured_output(MathSolution)

            response = structured_model.generate_content(
                f"Resuelve el siguiente problema matemático paso a paso y devuelve JSON estructurado: {user_content}"
            )

            # Convertir respuesta a JSON
            try:
                raw_json_text = response.text.strip()
                ai_data = json.loads(raw_json_text)

            except Exception as e:
                error_message = f"Error al interpretar JSON: {e}. Respuesta parcial: {raw_json_text[:200]}"

        except Exception as e:
            error_message = f"Error al generar respuesta del modelo: {e}"

    # Renderizar el template
    return render_template(
        'mathSolve/math_solve.html',
        ai_response=ai_data,
        error=error_message,
        user_content=user_content
    )