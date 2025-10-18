from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from controllers.plagiarism_controller import get_plagiarism_results

plagio_bp = Blueprint('plagio_bp', __name__)

@plagio_bp.before_request
def require_login():
    # Si el usuario NO está en la sesión (no ha iniciado sesión)
    if 'user_id' not in session:
        # Enviamos un mensaje de advertencia
        flash('Debes iniciar sesión para acceder a esta página.', 'warning')
        # Redirigimos al usuario a la página de login
        return redirect(url_for('auth_bp.login'))


@plagio_bp.route('/detectorPlagio/plagio-page', methods=['GET', 'POST'])
def plagio_page():
    return render_template('detectorPlagio/plagio_upload.html')

@plagio_bp.route('/upload', methods=['POST'])
def upload_file():
    return get_plagiarism_results()