

from transformers import pipeline

# Carga un modelo generativo pre-entrenado de Hugging Face
# El modelo 't5-small' es pequeño y rápido, ideal para pruebas. 
# Nota: La primera vez que se ejecute, descargará el modelo, lo cual puede tardar.
try:
    paraphrasing_model = pipeline("text2text-generation", model="t5-small")
except ImportError:
    print("Error: La librería 'transformers' o 'torch' no está instalada. Ejecuta 'pip install transformers torch'.")
    exit()

def paraphrase_text(text, max_length=500):
    """
    Toma un texto y devuelve una versión parafraseada.
    """
    try:
        # El modelo genera texto basándose en un 'prompt'
        paraphrased = paraphrasing_model(f"paraphrase: {text}", max_length)
        return paraphrased[0]['generated_text']
    except Exception as e:
        return f"Error al parafrasear el texto: {e}"

#if __name__ == '__main__':
    # Ejemplo de uso desde la terminal
    original_text = "El desarrollo de la tecnología de baterías de estado sólido marca un hito en la transición energética."
    paraphrased = paraphrase_text(original_text)
    print(f"Texto original: {original_text}")
    print(f"Texto parafraseado: {paraphrased}")