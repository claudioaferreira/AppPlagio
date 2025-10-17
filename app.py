# app.py
import os
import sys
from flask import Flask
from routes.routesAiTest import AiTest_bp
from routes.routesPlagio import plagio_bp
from routes.routesAi_detector import ai_detector_bp
from routes.routesParaphraser import paraphraser_bp
from sklearn.base import BaseEstimator, TransformerMixin 
from scipy.sparse import hstack 
import numpy as np 
import re 
from utils.ai_detectorLGBMClassifierFE import StylometricFeatureExtractor 


# en la ubicación correcta del entorno virtual (.venv).
project_root = os.path.dirname(os.path.abspath(__file__))

# La ruta a la carpeta donde se instalaron todas las librerías
site_packages_path = os.path.join(project_root, '.venv', 'Lib', 'site-packages')

# Agrega la ruta de site-packages a sys.path si no está ya
if site_packages_path not in sys.path:
    sys.path.append(site_packages_path)
    
# Agrega la raíz del proyecto para resolver las importaciones locales (routes/controllers)
if project_root not in sys.path:
    sys.path.append(project_root)

# >>> FIN DE LA INYECCIÓN DE PATH <<<

app = Flask(__name__)



# Opcional: Si quieres tener un manejador de archivos compartido
# Crea la carpeta de uploads si no existe
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Registrar los Blueprints
app.register_blueprint(plagio_bp)
app.register_blueprint(ai_detector_bp)
app.register_blueprint(paraphraser_bp)
app.register_blueprint(AiTest_bp)

if __name__ == '__main__':
    app.run(debug=True)