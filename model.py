import numpy as np
import pandas as pd
import pickle

data=pd.read_csv('/Users\saumy\will purchase.csv')
# print(data.head())

x=data.drop(['willpurchase'],axis=1)
y=data['willpurchase']
# print(y)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression

logmodel=LogisticRegression()

logmodel.fit(X_train,y_train)

print(logmodel.score(x,y))

print(logmodel.predict([[25,30000]]))

pickle.dump(logmodel,open('model.pkl','wb'))