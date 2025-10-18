from flask import Blueprint, render_template
from controllers.ai_controller import get_ai_results

ai_detector_bp = Blueprint('ai_detector_bp', __name__)


@ai_detector_bp.route('/detectorContenidoIA/ai-detector-page')
def ai_detector_page():
    return render_template('detectorContenidoIA/ai_detector.html')

@ai_detector_bp.route('/ai-detect', methods=['POST'])
def ai_detect():
    return get_ai_results()