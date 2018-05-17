# -*- coding: utf-8 -*-
"""
Created on Wed May  9 20:23:14 2018

@author: pgood
"""

from flask import Flask, render_template, request, jsonify, make_response, send_file, Markup

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('landing.html')


@app.route('/show_table', methods = ['POST'])
def show_best():
    from apply_model import make_table
    print('starting_model')
    table = make_table()
    html = Markup(table.to_html(index = False))
    return html

if __name__ == "__main__":
    app.run(host = '0.0.0.0')