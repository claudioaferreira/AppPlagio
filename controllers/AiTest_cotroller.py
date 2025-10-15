# controllers/AiTest_controller.py

from flask import render_template, request
from google import genai 
from google.genai.types import GenerateContentConfig 
from pydantic import BaseModel, Field #para JSON Structuring
from dotenv import load_dotenv
import os
import json

load_dotenv() 

# ==========================================================
# 1. ESQUEMA PYDANTIC (MODEL CONTEXT PROTOCOL DE ESTRUCTURA)
# ==========================================================
class MathSolution(BaseModel):
    """Esquema de salida para la solución matemática estructurada."""
    problema_identificado: str = Field(
        description="Breve reformulación y formalización matemática del problema dado, en español."
    )
    principios_clave: list[str] = Field(
        description="Lista de 3 a 5 teoremas o conceptos matemáticos principales que se aplicarán en la solución."
    )
    desarrollo_latex: str = Field(
        description="Desarrollo completo de la solución paso a paso. DEBE USAR ÚNICAMENTE NOTACIÓN LATEX CON $$Bloque$$."
    )
    respuesta_final: str = Field(
        description="El resultado final del problema, escrito en LaTeX (ej: $$x=4$$)."
    )

def test_ai_functionality():
    ai_data = None 
    error_message = None
    user_content = None

    # Mantiene la lógica de POST request
    if request.method == 'POST':
        user_content = request.form.get('content')
        
        # Lógica de validación
        if not user_content:
            error_message = "Por favor, introduce un problema matemático para generar una respuesta."
        else:
            try:
                # 2. Configuración de la API y Cliente
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("No se encontró la GOOGLE_API_KEY en el archivo .env")
                
                # --- CORRECCIÓN CLAVE: Inicialización del Cliente ---
                client = genai.Client(api_key=api_key)
                # ----------------------------------------------------

                # --- MODEL CONTEXT PROTOCOL (MCP) AVANZADO ---
                system_prompt = (
                    "ROL PROFESIONAL: Eres un Doctor en Matemáticas y Pedagogo. Tu único objetivo es generar "
                    "una respuesta COMPLETAMENTE en formato JSON que cumpla el esquema proporcionado (MathSolution). "
                    
                    "PROCESO DE PENSAMIENTO: Debes 1) Resolver con precisión. 2) Desglosar la solución. 3) Reforzar el concepto. "
                    
                    "RESTRICCIONES ESTRICTAS: "
                    "Si recibes otro mensaje que no sea un problema matemático, responde con un JSON vacio pero válido. Dejando saber que no es un problema matemático. "
                    "El contenido en los campos 'desarrollo_latex' y 'respuesta_final' DEBE UTILIZAR ESTRICTAMENTE "
                    "NOTACIÓN LATEX DE BLOQUE (doble dólar $$...$$) para todas las expresiones matemáticas. "
                    "No generes ningún texto fuera del objeto JSON."
                    "Si no puedes resolver el problema, explica brevemente por qué en 'problema_identificado' y "
                    "deja los otros campos vacíos."

                )
                # ----------------------------------------------------

                # 3. Creación del objeto de configuración para JSON Structuring
                config = GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=MathSolution,
                )

                # 4. Llamada a la API usando el Cliente
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=user_content,
                    config=config 
                )

                # 5. Procesamiento del JSON con limpieza defensiva
                raw_json_text = response.text.strip()
                
                if raw_json_text.startswith("```json"):
                    raw_json_text = raw_json_text[7:]
                if raw_json_text.endswith("```"):
                    raw_json_text = raw_json_text[:-3]
                
                ai_data = json.loads(raw_json_text.strip())
            
            except Exception as e:
                # Manejo de errores
                response_text_snippet = response.text[:100] if 'response' in locals() else 'No response received.'
                error_message = f"Ocurrió un error al procesar la respuesta: {e}. Respuesta parcial: {response_text_snippet}..."

    # Pasamos los datos al template (ai_data si es POST exitoso, o None)
    return render_template(
        'ai_test.html', 
        ai_response=ai_data,
        error=error_message, 
        user_content=user_content
    )