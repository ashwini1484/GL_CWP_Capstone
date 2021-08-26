import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle
import os

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('CWP Prediction Form.html')

# prediction function 
def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1, 11)
    loaded_model = pickle.load(open("CWP_flask.pkl", "rb"))
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST'])
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Win'
        else: 
            prediction ='Lose'
        return render_template("result.html", prediction = prediction)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True,port=port)
    app.config['TEMPLATES_AUTO_RELOAD'] = True
