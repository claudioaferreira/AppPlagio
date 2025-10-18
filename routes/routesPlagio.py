from flask import Flask, render_template,Blueprint, request
from controllers.plagiarism_controller import get_plagiarism_results

plagio_bp = Blueprint('plagio_bp', __name__)

@plagio_bp.route('/')
def welcome():
    return render_template('index.html')

@plagio_bp.route('/detectorPlagio/plagio-page', methods=['GET', 'POST'])
def plagio_page():
    return render_template('detectorPlagio/plagio_upload.html')

@plagio_bp.route('/upload', methods=['POST'])
def upload_file():
    return get_plagiarism_results()