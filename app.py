# app.py
import os
import sys
from flask import Flask
from routes.routesPlagio import plagio_bp
from routes.routesAi_detector import ai_detector_bp
from routes.routesParaphraser import paraphraser_bp

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

if __name__ == '__main__':
    app.run(debug=True)