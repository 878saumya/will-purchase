# from urllib import request

from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np

app= Flask(__name__)
server=app.server

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   age=int(request.form.get('age'))
   salary=int(request.form.get('salary'))
   # input_query = np.array([[age,salary]])

   result = model.predict(np.array([age,salary]).reshape(1,2))

   if result[0]==1:
       return render_template('index.html',label=1)
   else :
       return render_template('index.html',label=-1)

   # if result==1:
   #     return "will purchase"
   # else:
   #     return "will not buy"
   #
   # print(logmodel.predict([['age','salary']]))
   # result=logmodel.predict([['age','salary']])[0]
   #
   # if result==1:
   #     return "will purchase"
   # else :
   #     return "will not purchase"





if __name__== '__main__':
    app.debug = True
    app.run()


