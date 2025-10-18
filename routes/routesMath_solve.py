# routes/routesAiTest.py

from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from controllers.math_solve_cotroller import test_ai_functionality

AiTest_bp = Blueprint('AiTest_bp', __name__)
@AiTest_bp.before_request
def require_login():
    if 'user_id' not in session:
        # Si no está logueado, lo saca al login
        return redirect(url_for('auth_bp.login'))

@AiTest_bp.route('/mathSolve/math_solve.html', methods=['GET', 'POST'])
def ai_test_page():
    # Llama al controlador para que él decida qué hacer (entregar menú o cocinar)
    return test_ai_functionality()