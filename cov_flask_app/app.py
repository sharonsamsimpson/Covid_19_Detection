#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/covid', methods=['POST', 'GET'])
def rcovid():
    return render_template('covid.html')

@app.route('/covid.html', methods=['POST', 'GET'])
def covid():
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    if prediction == 1:
        pred = "are likely to have COVID"
    elif prediction == 0:
        pred = "are not likely to have COVID"
    output = pred
    return render_template('covid.html', prediction_text='You {}'.format(output))

@app.route('/back', methods=['POST', 'GET'])
def back():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

