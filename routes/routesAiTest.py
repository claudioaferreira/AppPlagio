# routes/routesAiTest.py

from flask import Blueprint
from controllers.AiTest_cotroller import test_ai_functionality

AiTest_bp = Blueprint('AiTest_bp', __name__)

@AiTest_bp.route('/ai_test', methods=['GET', 'POST'])
def ai_test_page():
    # Llama al controlador para que él decida qué hacer (entregar menú o cocinar)
    return test_ai_functionality()