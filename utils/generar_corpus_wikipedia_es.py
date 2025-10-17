# -*- coding: utf-8 -*-
# ============================================================
# Script: generar_corpus_wikipedia_es.py
# Autor: Claudio Ferreira (adaptado por ChatGPT)
# Descripcion:
# Extrae artículos de Wikipedia en español usando Wikiextractor
# y los guarda como archivos .txt identificados como
# "wikipedia-documentXXXX.txt" con soporte para reanudar descargas.
# ============================================================

import os
import json
import unicodedata
from tqdm import tqdm
from wikiextractor.WikiExtractor import extract_pages_from_file

# ====== CONFIGURACION ======
DIRECTORIO_DUMP = "wikipediaDump/eswiki-latest-pages-articles.xml.bz2"  # Dump descargado
DIRECTORIO_SALIDA = "corpus_wikipedia_es"
MAX_ARTICULOS = 1000        # Cantidad de artículos a guardar por ejecucion
ARCHIVO_PROGRESO = "progreso.json"  # Guarda el índice alcanzado
# ============================

os.makedirs(DIRECTORIO_SALIDA, exist_ok=True)

# ====== Funcion para cargar progreso ======
def cargar_progreso():
    if os.path.exists(ARCHIVO_PROGRESO):
        with open(ARCHIVO_PROGRESO, "r", encoding="utf-8") as f:
            datos = json.load(f)
            return datos.get("ultimo_indice", 0)
    return 0

# ====== Funcion para guardar progreso ======
def guardar_progreso(indice):
    with open(ARCHIVO_PROGRESO, "w", encoding="utf-8") as f:
        json.dump({"ultimo_indice": indice}, f, indent=2)

# ====== Iteracion principal ======
print("Extrayendo artículos de Wikipedia en español...")
inicio = cargar_progreso()
contador = inicio

for title, text in tqdm(extract_pages_from_file(DIRECTORIO_DUMP), desc="Extrayendo artículos"):
    if contador >= inicio + MAX_ARTICULOS:
        break

    texto = unicodedata.normalize("NFKC", text.strip())
    if not texto:
        continue

    nombre_archivo = f"wikipedia-document{contador+1:05d}.txt"
    ruta = os.path.join(DIRECTORIO_SALIDA, nombre_archivo)

    with open(ruta, "w", encoding="utf-8") as f:
        f.write(f"Título: {title}\n\n")
        f.write(texto)

    contador += 1

    # Guardar progreso cada 10 artículos
    if contador % 10 == 0:
        guardar_progreso(contador)

guardar_progreso(contador)
print("\n✅ Corpus generado correctamente.")
print(f"Archivos almacenados en: {os.path.abspath(DIRECTORIO_SALIDA)}")
print(f"Se guardaron artículos del {inicio+1} al {contador}.")
print(f"Puedes ejecutar el script nuevamente para continuar desde el {contador+1}.")
