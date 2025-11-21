import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Configuramos el modelo
model_id = "ibm-granite/granite-4.0-h-1b"

#Cargamos el modelo y el tokenizador
#print(f"Cargando el modelo {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Cargamos el modelo y que use bfloat16 para ahorrar memoria
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Si hay una GPU disponible, movemos el modelo a la GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
#print(f"Usando dispositivo: {device}")

# Plantilla base del prompt
PROMPT_TEMPLATES = {
    # --- Tonos Gratuitos ---
    "estandar": """Instrucción: Parafrasea el siguiente texto en español, manteniendo el mismo significado y un tono profesional estándar.

Texto: {text}

Parafraseo:""",

    "simplificado": """Instrucción: Simplifica el siguiente texto en español. Hazlo muy fácil de entender, usando palabras comunes y frases cortas, como para alguien que no conoce el tema.

Texto: {text}

Parafraseo:""",

    "fluido": """Instrucción: Reescribe el siguiente texto en español para que sea más fluido y natural, mejorando la redacción y la conexión entre ideas sin cambiar el significado.

Texto: {text}

Parafraseo:""",

    "abogado": """Instrucción: Parafrasea el siguiente texto en español con un tono legal. Sé extremadamente preciso, formal y cauto con las palabras. El resultado debe sonar como un documento o consejo legal.

Texto: {text}

Parafraseo:""",

    "casual": """Instrucción: Reescribe el siguiente texto en español con un tono casual y relajado. Usa un lenguaje informal y conversacional, como si estuvieras hablando.

Texto: {text}

Parafraseo:""",

    "bachillerato": """Instrucción: Reescribe el siguiente texto en español como si estuviera dirigido a estudiantes de bachillerato. Usa un tono claro, educativo y fácil de seguir, pero sin perder la corrección gramatical.

Texto: {text}

Parafraseo:""",

    # --- Tonos Premium ---
    "creativo": """Instrucción: Reescribe el siguiente texto en español de forma más creativa, original e imaginativa. No tengas miedo de usar metáforas o un lenguaje más vívido si ayuda.

Texto: {text}

Parafraseo:""",

    "ingeniero": """Instrucción: Parafrasea el siguiente texto en español. Utiliza un lenguaje técnico, preciso y formal, similar al de un ingeniero o un documento técnico. Enfócate en la funcionalidad y la lógica.

Texto: {text}

Parafraseo:""",

    "doctor": """Instrucción: Parafrasea el siguiente texto en español con un tono médico profesional. Usa terminología clínica apropiada y mantén un tono formal y empático.

Texto: {text}

Parafraseo:""",

    "formal": """Instrucción: Reescribe el siguiente texto en español usando un tono muy formal y educado. Evita contracciones y utiliza un vocabulario elevado.

Texto: {text}

Parafraseo:""",

    "amistoso": """Instrucción: Reescribe el siguiente texto en español con un tono cercano, cálido y amistoso. Usa un lenguaje positivo y accesible, como si hablaras con un amigo.

Texto: {text}

Parafraseo:""",

    "profesional": """Instrucción: Parafrasea el siguiente texto en español para un entorno de negocios. Sé directo, claro, cortés y enfocado a resultados.

Texto: {text}

Parafraseo:""",

    "diplomatico": """Instrucción: Reescribe el siguiente texto en español con un tono diplomático. Sé cortés, neutral y evita cualquier lenguaje que pueda parecer confrontativo o emocional.

Texto: {text}

Parafraseo:""",

    "seguro": """Instrucción: Reescribe el siguiente texto en español con un tono firme y seguro. Transmite confianza, liderazgo y dominio del tema sin sonar arrogante.

Texto: {text}

Parafraseo:""",

    "persuasivo": """Instrucción: Reescribe el siguiente texto en español con un tono persuasivo. El objetivo es convencer al lector de una idea o para que tome una acción.

Texto: {text}

Parafraseo:""",

    "secundaria": """Instrucción: Reescribe el siguiente texto en español adaptándolo para estudiantes de secundaria. Usa oraciones cortas, lenguaje sencillo y ejemplos cuando sea posible.

Texto: {text}

Parafraseo:""",

    "academico": """Instrucción: Reescribe el siguiente texto en español con un tono académico. Utiliza un lenguaje formal, preciso y estructurado, como el de un artículo de investigación o un ensayo universitario.

Texto: {text}

Parafraseo:""",

    "vivido": """Instrucción: Reescribe el siguiente texto en español con un tono vívido y descriptivo. Usa un lenguaje sensorial, colorido y expresivo que ayude al lector a visualizar la escena o idea.

Texto: {text}

Parafraseo:""",

    "empatico": """Instrucción: Reescribe el siguiente texto en español con un tono empático y humano. Muestra comprensión y sensibilidad hacia las emociones del lector.

Texto: {text}

Parafraseo:""",

    "lujoso": """Instrucción: Reescribe el siguiente texto en español con un tono elegante y lujoso. Usa un lenguaje refinado, sofisticado y con un ritmo pausado que refleje exclusividad.

Texto: {text}

Parafraseo:""",

    "atractivo": """Instrucción: Reescribe el siguiente texto en español con un tono atractivo y cautivador. Haz que suene interesante y mantenga la atención del lector.

Texto: {text}

Parafraseo:""",

    "directo": """Instrucción: Reescribe el siguiente texto en español con un tono directo y conciso. Ve al punto sin rodeos, pero manteniendo la cortesía.

Texto: {text}

Parafraseo:""",

    # --- Fallback (Obligatorio) ---
    "default": """Instrucción: Parafrasea el siguiente texto en español, manteniendo el mismo significado y un tono profesional.

Texto: {text}

Parafraseo:"""
}

# Función para parafrasear texto
def parafreasearTexto(original_text: str, tone: str = "estandar") -> str:
    print(f"TONO SELECCIONADO: {tone}")
    print(f"Iniciando parafraseo para: \"{original_text[:50]}...\"")
    selected_template = PROMPT_TEMPLATES.get(tone, PROMPT_TEMPLATES["default"])
    print(f"Iniciando parafraseo (Tono: {tone}) para: \"{original_text[:50]}...\"")

    try:
        # --- 3. Crear el Prompt (La Instrucción) ---
        prompt = selected_template.format(text=original_text)

        # --- 4. Generar el Texto ---
        # Tokenizamos
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        print("Generando texto...")
        # Generamos la salida
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

        # --- 5. Decodificar y Extraer el Resultado ---
        resultado_completo = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extraemos solo el texto parafraseado
        
        secciones = resultado_completo.split("Parafraseo:")
        if len(secciones) > 1:
            # Tomamos lo que sigue a "Parafraseo:" y lo dividimos por parrafos vacios,
            # quedandonos solo con el primero.
            texto_parafraseado = secciones[1].split("\n\n")[0].strip()
        else:
            # Fallback por si "Parafraseo:" no aparece
            texto_parafraseado = resultado_completo
        
        print(f"Parafraseo generado: \"{texto_parafraseado[:50]}...\"")
        return texto_parafraseado

    except IndexError:
        print("Error: No se pudo encontrar 'Parafraseo:' en la salida del modelo.")
        return "Error al generar el parafraseo (formato de salida inesperado)."
    except Exception as e:
        print(f"Error durante la generación de parafraseo: {e}")
        return f"Error al procesar el texto: {e}"

# --- prueba ---
if __name__ == "__main__":
    test_text = "La inteligencia artificial está cambiando la manera en que las personas trabajan, estudian y se comunican. Cada día surgen nuevas aplicaciones capaces de analizar datos, automatizar tareas y ofrecer soluciones precisas en segundos. Sin embargo, su uso también plantea desafíos éticos sobre la privacidad, el empleo y la toma de decisiones humanas. Por eso, es fundamental aprender a usarla con responsabilidad y criterio."
    
    print("\n--- INICIANDO PRUEBA LOCAL ---")
    paraphrased = parafreasearTexto(test_text)
    
    print("\n--- TEXTO ORIGINAL DE PRUEBA ---")
    print(test_text)
    
    print("\n--- TEXTO PARAFRASEADO DE PRUEBA ---")
    print(paraphrased)
    
    print("--- FIN DE PRUEBA LOCAL ---")