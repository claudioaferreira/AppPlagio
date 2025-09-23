# controllers/AiTest_controller.py

from flask import render_template, request
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv() # Carga las variables del archivo .env

def test_ai_functionality():
    if request.method == 'POST':
        user_content = request.form.get('content')
        ai_response = None
        error_message = None

        # Asegúrate de que el contenido del usuario no esté vacío
        if not user_content:
            error_message = "Por favor, introduce un texto para generar una respuesta."
        else:
            try:
                # --- CAMBIO IMPORTANTE ---
                # 1. Configura la API key directamente. Es más explícito y confiable.
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("No se encontró la GOOGLE_API_KEY en el archivo .env")
                
                genai.configure(api_key=api_key)

                system_prompt = (
                    "Eres un tutor de matemáticas avanzado. Tu propósito es resolver problemas matemáticos "
                    "y explicar los conceptos detrás de ellos de manera clara y paso a paso. "
                    "Utiliza siempre la notación LaTeX para todas las ecuaciones y expresiones matemáticas, "
                    "encerrándolas entre signos de dólar ($...$ para texto en línea y $$...$$ para bloques de ecuaciones). "
                    "Sé preciso, lógico y didáctico en todas tus respuestas y siempre responde en espanol."
                )

                # 2. Pasa la instrucción al crear el modelo usando el parámetro 'system_instruction'.
                model = genai.GenerativeModel(
                    'gemini-1.5-flash-latest',
                    system_instruction=system_prompt
                )

                # 3. Se genera el contenido desde la instancia del modelo.
                response = model.generate_content(user_content)
                
                ai_response = response.text
            
            except Exception as e:
                # El error ahora será más descriptivo.
                error_message = f"Ocurrió un error al contactar la IA: {e}"

        return render_template('ai_test.html', ai_response=ai_response, error=error_message, user_content=user_content)

    # Para peticiones GET, simplemente muestra la página.
    return render_template('ai_test.html')