from flask import render_template, request
from utils.paraphraser import paraphrase_text

def get_paraphraser_page():
    return render_template('paraphraser.html')

def get_paraphrased_result():
    original_text = request.form['original_text']
    paraphrased_text = paraphrase_text(original_text)
    
    return render_template('paraphrased_result.html',
                           original_text=original_text,
                           paraphrased_text=paraphrased_text)