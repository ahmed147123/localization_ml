from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/getposition',methods=['POST','GET'])
def get_position():
    if request.method=='POST':
        result=request.form
        headers = [ "STUDBME1","STUDBME2", "ARC3"]
        d = {headers[0]:result["STUDBME1"],headers[1]:result["STUDBME2"] ,headers[2]:result["ARC3"]}
        df = pd.DataFrame(data=d, index=[0])
        
        pkl_file = open('savedModel.pkl', 'rb')
        logmodel = pickle.load(pkl_file)
        prediction = logmodel.predict(df)
        
        return render_template('result.html',prediction=prediction)

    
if __name__ == '__main__':
	app.run()


