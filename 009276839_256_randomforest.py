#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:41:33 2017

@author: AaronNguyen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('dota2Train.csv')
testset = pd.read_csv('dota2Test.csv')

# Selecting valuable attributes
y_train = dataset.iloc[:,0].values
X_train = dataset.iloc[:,4:].values
#y = y[:, None]
y_test = testset.iloc[:,0].values
X_test = testset.iloc[:,4:].values

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_features= 'log2', n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print (cm)

# Calculating Accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Calculating F1 score
from sklearn.metrics import f1_score
f1 = f1_score(y_test,y_pred, average = 'macro')
print(f1)


    
    
    


