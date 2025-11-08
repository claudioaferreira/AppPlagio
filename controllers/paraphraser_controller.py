from flask import render_template, request
from utils.paraphraser import paraphrase_text
from utils.parafraseadorGranite import parafreasearTexto

def get_paraphraser_page():
    return render_template('/parafraseador/paraphraser.html')

def get_paraphrased_result():
    original_text = request.form['original_text']
    paraphrased_text = parafreasearTexto(original_text)
    
    return render_template('parafraseador/paraphrased_result.html',
                           original_text=original_text,
                           paraphrased_text=paraphrased_text)