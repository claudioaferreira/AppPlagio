

from transformers import pipeline

# Carga un modelo generativo pre-entrenado de Hugging Face
# El modelo 't5-small' es pequeño y rápido, ideal para pruebas. t5-base * humarin/t5-small-finetuned-paraphrase / google/flan-t5-base
# Nota: La primera vez que se ejecute, descargará el modelo, lo cual puede tardar.
try:
    paraphrasing_model = pipeline("text2text-generation", model="google/flan-t5-base")
except ImportError:
    print("Error: La librería 'transformers' o 'torch' no está instalada. Ejecuta 'pip install transformers torch'.")
    exit()

def paraphrase_text(text, max_length=500):
    """
    Toma un texto y devuelve una versión parafraseada.
    """
    try:
        # El modelo genera texto basándose en un 'prompt añadiendo el spanish'
        #prompt que se pueden usar: paraphrase in Spanish: {texto}, simplify in Spanish: {texto}, expand in Spanish: {texto}, rewrite this in a professional tone in Spanish: {texto}
        #rewrite this in a simple tone in Spanish: {texto},rewrite this for a general audience in Spanish: {texto}
        paraphrased = paraphrasing_model(f"paraphrase in Spanish: {text}", max_new_tokens=800)
        return paraphrased[0]['generated_text']
    except Exception as e:
        return f"Error al parafrasear el texto: {e}"

#if __name__ == '__main__':
    # Ejemplo de uso desde la terminal
    original_text = "El desarrollo de la tecnología de baterías de estado sólido marca un hito en la transición energética."
    paraphrased = paraphrase_text(original_text)
    print(f"Texto original: {original_text}")
    print(f"Texto parafraseado: {paraphrased}")