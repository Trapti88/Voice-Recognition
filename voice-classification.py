# Author = TRAPTI MESHRAM(aliya)
# Date = 01/12/2022

"""This is  voice Classification project on 
 Male and Female sound
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.model_selection import GridSearchCV
import seaborn as sn

# first load data

df = pd.read_csv('C:/Users/DELL/Documents/Voice/voice.csv')
#print(df)
#df.info() # show rows and colums and enteries = 3168

df.isnull().sum()
print("shape of Data :" , df.shape)
print("Total number of labels:{}".format(df.shape[0]))
print("Number of male:{}".format(df[df.label =='male'].shape[0]))
print("Number of female:{}".format(df[df.label =='female'].shape[0]))

X = df.iloc[:,:-1]
print(df.shape)
print(X.shape)

# change text in  0 or 1
y = df.iloc[:,:-1]
gender = LabelEncoder()
y = gender.fit_transform(y)
print(y)

# Scaling data
scaler = StandardScaler()
scaler.fit(X)
x = scaler.transform(X)

# test and train data
X_train,Y_train,x_test,y_test = train_test_split(X,y,test_size =0.3, random_state= 100)

# svc model
svc_model = SVC()
svc_model.fit(X_train, Y_train)
y_pred = svc_model.predict(x_test)
print('Accuracy Score :')
print(metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test,y_pred))

# Grid search
param_grid = {'C': [0.1,1,10,100] , 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, refit =True, verbose =2)
grid.fit(X_train, Y_train)
grid_predi=grid.predict(x_test)
print('Accuracy Score :')(metrics.accuracy_score(y_test,grid_predi))
print(confusion_matrix(y_test, grid_predi))
print(classification_report(y_test, grid_predi))

## End