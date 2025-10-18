from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from controllers.paraphraser_controller import get_paraphraser_page, get_paraphrased_result

paraphraser_bp = Blueprint('paraphraser_bp', __name__)

@paraphraser_bp.before_request
def require_login():
    if 'user_id' not in session:
        # Si no está logueado, lo saca al login
        return redirect(url_for('auth_bp.login'))

@paraphraser_bp.route('/parafraseador/paraphraser-page')
def paraphraser_page():
    return get_paraphraser_page()

@paraphraser_bp.route('/paraphrase', methods=['POST'])
def paraphrase():
    return get_paraphrased_result()