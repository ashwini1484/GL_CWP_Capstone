import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle
import os
from sklearn.preprocessing import StandardScaler
import pprint

app= Flask(__name__)


@app.route('/')
def home():
    return render_template('CWP Prediction Form.html')

def list_dump(list_write,fil):
   f = open(fil+'.txt', 'w')
   for ele in list_write:
       f.write(str(ele) + '\n')

   f.close()
   return

# prediction function 
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 21)
    #ss=pickle.load(open("CWP_flask3.pkl","rb"))
    #to_predict_scale = ss.transform(to_predict)
    #list_dump(to_predict_scale,'f2')
    loaded_model = pickle.load(open("CWP_flask2.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
  
@app.route('/result', methods = ['POST'])
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list))
        list_dump(to_predict_list,'f1')
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
