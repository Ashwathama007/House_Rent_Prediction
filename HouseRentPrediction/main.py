from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np
df=pd.read_csv("Bengaluru_Mod_Prj.csv")
pipe=pickle.load(open("RidgeModel.pkl",'rb'))

app = Flask(__name__)
@app.route('/')
def index():
    locations=sorted(df['location'].unique())
    return render_template('index.html',locations=locations)

@app.route('/predict',methods=['POST'])
def predict():
    location=request.form.get('location')
    bhk=request.form.get('bhk')
    bath=request.form.get('bath')
    sqft=request.form.get('size')
    #print(bhk,location,bath,sqft)
    input=pd.DataFrame([[location,sqft,bath,bhk]],columns=['location','total_sqft','bath','BHK'])
    prediction=pipe.predict(input)[0]*1e5
    return str(np.round(prediction,2))
if __name__=="__main__":
    app.run(debug=True,port=5000)

