# 🛡️ AppPlagio: Suite de Inteligencia Artificial para Análisis de Texto

### 1. Detector de Plagio Semántico (Enfoque Granular)
A diferencia de los detectores tradicionales que buscan coincidencias exactas, este módulo entiende el significado del texto.
- **Tecnología:** Utiliza modelos de embeddings (`stsb-xlm-r-multilingual` y `paraphrase-multilingual-mpnet-base-v2`) para transformar frases en vectores.
- **Capacidad:** Detecta plagio por paráfrasis (textos reescritos con otras palabras pero mismo significado).
- **Método:** Análisis granular por oraciones y comparación de similitud del coseno contra una base de datos vectorial en SQL Server.

### 2. Detector de Contenido Generado por IA
Clasifica si un texto fue escrito por un humano o por modelos de lenguaje (ChatGPT, Gemini, Llama, etc.).
- **Modelo:** **LightGBM** (Light Gradient Boosting Machine) optimizado para alta precisión.
- **Análisis:** Utiliza **TF-IDF** con n-gramas (1-3) para analizar patrones lingüísticos y secuencias de palabras, generando un "AI Score" de probabilidad.
- **Clases:** Identifica desde "Completamente humano" hasta "Texto profundamente mezclado" o "Editado por máquina".


### 3. Parafraseador Multi-Tono (GenAI)
Reescritura inteligente de textos manteniendo el significado original pero cambiando el estilo.
- **Motor:** **IBM Granite** (`ibm-granite/granite-4.0-h-1b`), un modelo causal robusto.
- **Estilos:** Soporta más de 20 tonos, incluyendo: *Legal, Médico, Académico, Creativo, Simplificado, Agresivo, etc.*

### 4. Math Solver (Razonamiento Matemático)
Resuelve problemas matemáticos complejos mostrando el procedimiento y la lógica.
- **Motor:** API de **Google Gemini** (`gemini-2.5-flash`).
- **Formato:** Salida estructurada con renderizado **LaTeX** para fórmulas matemáticas precisas.

---

## 🛠 Stack Tecnológico

- **Lenguaje:** Python 3.8+
- **Backend Framework:** Flask
- **Base de Datos:** SQL Server 2022 (Almacenamiento de vectores `VARBINARY`)
- **Inteligencia Artificial:**
  - `sentence-transformers` (Embeddings)
  - `scikit-learn` (Preprocesamiento y métricas)
  - `lightgbm` (Clasificación IA)
  - `transformers` (Hugging Face - Inferencia local)
  - `google-genai` (API Cloud)
  - `nltk` (Procesamiento de lenguaje natural)

---

---
## 🚀 Cómo Poner en Marcha el Proyecto

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Requisitos Previos
Asegúrate de que tu máquina tenga instaladas las siguientes herramientas:
- **Python 3.8+**: El lenguaje principal del proyecto.
- **Gestor de Bases de Datos SQL Server 2022**: La base de datos donde se almacenarán los embeddings.
- **Controlador ODBC para SQL Server**: Necesario para que Python se conecte a la base de datos.
- **Git**: Para clonar el repositorio.

---

### 2. Configuración del Entorno de Python
Clona el repositorio:
```bash
git clone [[URL_DEL_REPOSITORIO](https://github.com/claudioaferreira/AppPlagio)]
cd [AppPlagio]
```
Crea y activa un entorno virtual:
```bash
python -m venv venv
# En Windows
venv\Scripts\activate
```
Instala las dependencias:
```bash
pip install -r requirements.txt
```
---

### 3. Configuración de la Base de Datos
   
- Habilita la autenticación de SQL Server:
- Abre SQL Server Management Studio (SSMS) y conéctate usando la autenticación de Windows.
- Haz clic derecho en tu servidor > Propiedades > Seguridad.
- Selecciona "Modo de autenticación de SQL Server y Windows".
- Reinicia el servicio de SQL Server.
- Crea el usuario sa y la base de datos:
- Crea una base de datos llamada plagioEmb.
- Asegúrate de que el usuario sa esté habilitado y tenga permisos de db_owner sobre esta base de datos.
- Crea la tabla para los embeddings:
- Ejecuta el siguiente script en tu base de datos plagioEmb:
```bash
CREATE TABLE DocumentEmbeddings (
    DocumentID NVARCHAR(255) PRIMARY KEY,
    DocumentText NVARCHAR(MAX),
    Embedding VARBINARY(MAX)
);
```
---

### 5. Descarga y Configuración de los Datasets
El proyecto utiliza dos datasets principales. Debes descargarlos y colocarlos en carpetas específicas en la raíz de tu proyecto.

- Detección de Plagio Extrínseco (Corpus PAN 2011):
Descarga el corpus ([aqui](https://www.google.com/search?q=https://pan.webis.de/clef11/pan11-web/plagiarism-detection.html)).
- Descomprime el archivo y coloca la carpeta external-detection-corpus dentro de una nueva carpeta llamada Pan-Plagiarism-Corpus-2011.
- Detección de IA (Corpus PAN 2025):
Solicita acceso al corpus ([aqui](https://www.google.com/search?q=https://pan.webis.de/clef25/pan25-web/ai-generated-text-detection.html)) llenando el formulario.
- Una vez que lo tengas, descomprime el archivo y coloca los archivos .jsonl en una nueva carpeta llamada pan-25-ai-detection.
  
  ---

### 6. Ejecutar la Aplicación
- Cargar el Corpus de Plagio en la DB:
- Abre generate_embeddingsSQL.py y configura tus credenciales de SQL Server.
- Ejecuta el script: python generate_embeddingsSQL.py
- Este proceso es de una sola vez y puede tardar varios minutos en cargar los embeddings en la base de datos.
- Ejecutar la App Principal:
- Abre app.py y configura las mismas credenciales de SQL Server.
- Corre el servidor de Flask: 

```bash 
python app.py
```
- Usar la Aplicación:
- Abre tu navegador y ve a http://127.0.0.1:5000.
- El modelo de detección de IA se entrenará automáticamente la primera vez que accedas a la página del detector de IA, por lo que la primera carga será más lenta.

---

## 🛠 Tecnologías Utilizadas

- **Backend:** Python con Flask.
- **Inteligencia Artificial:**
    - **Embeddings:** sentence-transformers para análisis semántico.
    - **Clasificación:** scikit-learn y TF-IDF para detectar texto generado por IA.
- **Base de Datos:** SQL Server 2022 con capacidades de búsqueda de vectores.
- **Frontend:** HTML y CSS con una interfaz de usuario moderna.
- **Gestión de Datos:** pandas y joblib.

---

## ⚠️ Resolución de Problemas
- Si tienes errores de conexión a la base de datos, revisa que tus credenciales en generate_embeddingsSQL.py y app.py sean correctas y que el usuario sa esté habilitado.
- Si no ves el style.css, verifica que el archivo esté en una carpeta llamada static en la raíz de tu proyecto.
- Si el modelo de IA no se entrena, asegúrate de que los archivos .jsonl del corpus PAN 2025 estén en la carpeta correcta.