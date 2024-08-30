from flask import Flask,render_template,request,jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

#import ridge regressor and standard scaler pickle
ridge_model=pickle.load(open('Model/ridge.pkl','rb'))
standard_scaler=pickle.load(open('Model/scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction',methods=['GET','POST'])
def predict_datapoint():
        return render_template('home.html')
        
@app.route('/results',methods=['GET','POST'])
def results():
        if request.method=='POST':
              temp=float(request.form.get('Temperature'))
              RH=float(request.form.get('RH'))
              Ws=float(request.form.get('Ws'))
              RAIN=float(request.form.get('RAIN'))
              FFMC=float(request.form.get('FFMC'))
              DMC=float(request.form.get('DMC'))
              ISI=float(request.form.get('ISI'))
              Classes=float(request.form.get('Classes'))
              Region=float(request.form.get('Region'))

              scaled_input=standard_scaler.transform([[temp,RH,Ws,RAIN,FFMC,DMC,ISI,Classes,Region]])
              result=ridge_model.predict(scaled_input)

              return render_template('result.html',Result=result[0])
        else:
              return "ERROR!!"

if __name__=='__main__':
    app.run(debug=True)