from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from controllers.ai_controller import get_ai_results

ai_detector_bp = Blueprint('ai_detector_bp', __name__)

@ai_detector_bp.before_request
def require_login():
    if 'user_id' not in session:
        # Si no está logueado, lo saca al login
        return redirect(url_for('auth_bp.login'))

@ai_detector_bp.route('/detectorContenidoIA/ai-detector-page')
def ai_detector_page():
    return render_template('detectorContenidoIA/ai_detector.html')

@ai_detector_bp.route('/ai-detect', methods=['POST'])
def ai_detect():
    return get_ai_results()