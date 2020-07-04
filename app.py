
from flask import Flask,render_template,request
import joblib
import numpy as np
import pandas as pd

app=Flask(__name__)
model = joblib.load('model/model.pickle')

@app.route('/')
def home():
    return render_template('form.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    gender=request.form.get("gender")
    following=request.form.get("following")
    follower=request.form.get("follower")
    totaltweet=request.form.get("totaltweet")
               
    baru=np.array([gender, following, follower,totaltweet]).reshape((1,-1))
    x_new = model.predict(baru)
        
        
    return render_template('form.html', prediction_text='This Person Will Become {}'.format(x_new[0]))

@app.route('/predict_api',methods=['POST'])

def predict_api():
    '''
    For direct API calls trought request
    '''
    gender=request.form.get("gender")
    following=request.form.get("following")
    follower=request.form.get("follower")
    totaltweet=request.form.get("totaltweet")
               
    baru=np.array([gender, following, follower,totaltweet]).reshape((1,-1))
    x_new = model.predict(baru)
    return jsonify(x_new[0])
if __name__ == "__main__":
    app.run()
