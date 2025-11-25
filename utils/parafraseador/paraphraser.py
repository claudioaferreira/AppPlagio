import gc
from transformers import pipeline

# Variable global
_paraphrasing_pipeline = None

def get_paraphraser_model():
    global _paraphrasing_pipeline
    if _paraphrasing_pipeline is None:
        print("Cargando modelo de parafraseo en RAM...")
        _paraphrasing_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
    return _paraphrasing_pipeline

def unload_paraphraser_model():
    global _paraphrasing_pipeline
    if _paraphrasing_pipeline is not None:
        print("Liberando memoria del modelo de parafraseo...")
        del _paraphrasing_pipeline
        _paraphrasing_pipeline = None
        gc.collect() # Forza a Python a limpiar la RAM inmediatamente

def paraphrase_text(text, max_length=500):
    try:
        model = get_paraphraser_model()
        paraphrased = model(f"paraphrase in Spanish: {text}", max_new_tokens=800)
        result = paraphrased[0]['generated_text']
        
        # OPCIÓN A: Descargar inmediatamente después de usar (Ahorro máximo de RAM)
        unload_paraphraser_model() 
        
        return result
    except Exception as e:
        return f"Error al parafrasear: {e}"

#if __name__ == '__main__':
    # Ejemplo de uso desde la terminal
    original_text = "El desarrollo de la tecnología de baterías de estado sólido marca un hito en la transición energética."
    paraphrased = paraphrase_text(original_text)
    print(f"Texto original: {original_text}")
    print(f"Texto parafraseado: {paraphrased}")