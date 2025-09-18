# Detector de Plagio y Contenido de IA 📝


Este proyecto es una aplicación web avanzada para la validación de la originalidad del texto. Utiliza un enfoque de análisis semántico para entender el significado del texto, lo que le permite identificar plagio por paráfrasis. Además, incluye un segundo componente que clasifica si un texto fue escrito por un humano o por una inteligencia artificial.

---
##🚀 Cómo Poner en Marcha el Proyecto

Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

### 1. Requisitos Previos
Asegúrate de que tu máquina tenga instaladas las siguientes herramientas:
- **Python 3.8+**: El lenguaje principal del proyecto.
- **Gestor de Bases de Datos SQL Server 2022**: La base de datos donde se almacenarán los embeddings.
- **Controlador ODBC para SQL Server**: Necesario para que Python se conecte a la base de datos.
- **Git**: Para clonar el repositorio.

### 2. Configuración del Entorno de Python
Clona el repositorio:
```bash
git clone [URL_DEL_REPOSITORIO]
cd [nombre-del-repositorio]
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
Tu proyecto utiliza dos datasets principales. Debes descargarlos y colocarlos en carpetas específicas en la raíz de tu proyecto.

- Detección de Plagio Extrínseco (Corpus PAN 2011):
Descarga el corpus ([aqui](https://www.google.com/search?q=https://pan.webis.de/clef11/pan11-web/plagiarism-detection.html)).
- Descomprime el archivo y coloca la carpeta external-detection-corpus dentro de una nueva carpeta llamada Pan-Plagiarism-Corpus-2011.
- Detección de IA (Corpus PAN 2025):
Solicita acceso al corpus ([aqui](https://www.google.com/search?q=https://pan.webis.de/clef25/pan25-web/ai-generated-text-detection.html)) llenando el formulario.
- Una vez que lo tengas, descomprime el archivo y coloca los archivos .jsonl en una nueva carpeta llamada pan-25-ai-detection.
  
### 6. Ejecutar la Aplicación
- Cargar el Corpus de Plagio en la DB:
- Abre generate_embeddingsSQL.py y configura tus credenciales de SQL Server.
- Ejecuta el script: python generate_embeddingsSQL.py
- Este proceso es de una sola vez y puede tardar varios minutos en cargar los embeddings en la base de datos.
- Ejecutar la App Principal:
- Abre app.py y configura las mismas credenciales de SQL Server.
- Corre el servidor de Flask: python app.py
- Usar la Aplicación:
- Abre tu navegador y ve a http://127.0.0.1:5000.
- El modelo de detección de IA se entrenará automáticamente la primera vez que accedas a la página del detector de IA, por lo que la primera carga será más lenta.

## 🛠 Tecnologías Utilizadas

Backend: Python con Flask.
Inteligencia Artificial:
Embeddings: sentence-transformers para análisis semántico.
Clasificación: scikit-learn y TF-IDF para detectar texto generado por IA.
Base de Datos: SQL Server 2022 con capacidades de búsqueda de vectores.
Frontend: HTML y CSS con una interfaz de usuario moderna.
Gestión de Datos: pandas y joblib.


## ⚠️ Resolución de Problemas
Si tienes errores de conexión a la base de datos, revisa que tus credenciales en generate_embeddingsSQL.py y app.py sean correctas y que el usuario sa esté habilitado.
Si no ves el style.css, verifica que el archivo esté en una carpeta llamada static en la raíz de tu proyecto.
Si el modelo de IA no se entrena, asegúrate de que los archivos .jsonl del corpus PAN 2025 estén en la carpeta correcta.

