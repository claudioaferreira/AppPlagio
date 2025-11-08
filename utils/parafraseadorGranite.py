import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Configuramos el modelo
model_id = "ibm-granite/granite-4.0-h-1b"

#Cargamos el modelo y el tokenizador
print(f"Cargando el modelo {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

#Cargamos el modelo y que use bfloat16 para ahorrar memoria
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Si hay una GPU disponible, movemos el modelo a la GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Usando dispositivo: {device}")

# Plantilla base del prompt
base_prompt_template = """Instrucción: Parafrasea el siguiente texto en español, manteniendo el mismo significado y un tono profesional.

Texto: {text}

Parafraseo:"""

# Función para parafrasear texto
def parafreasearTexto(original_text: str) -> str:

    print(f"Iniciando parafraseo para: \"{original_text[:50]}...\"")

    try:
        # --- 3. Crear el Prompt (La Instrucción) ---
        prompt = base_prompt_template.format(text=original_text)

        # --- 4. Generar el Texto ---
        # Tokenizamos
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

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