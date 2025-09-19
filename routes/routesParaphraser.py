from flask import Blueprint
from controllers.paraphraser_controller import get_paraphraser_page, get_paraphrased_result

paraphraser_bp = Blueprint('paraphraser_bp', __name__)

@paraphraser_bp.route('/paraphraser-page')
def paraphraser_page():
    return get_paraphraser_page()

@paraphraser_bp.route('/paraphrase', methods=['POST'])
def paraphrase():
    return get_paraphrased_result()