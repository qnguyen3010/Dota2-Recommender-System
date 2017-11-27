#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 10:42:12 2017

@author: AaronNguyen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
pre_dataset = pd.read_csv('dota2Train.csv')

# Feature Selection
pre_dataset.drop(pre_dataset.columns[[1,2,3]],axis=1, inplace=True)
pre_matrix = pre_dataset.values

# Creating matrix with all winning results
matrix = []
for row in pre_matrix:
    if row[0] == 1:
        matrix.append(row[1:])
        
# Extracting first 5 users as sample of users
samp_user = []
for i in range(5):
    samp_user.append(matrix[i])

# Calculating cosine similarities
from sklearn.metrics.pairwise import cosine_similarity

# Finding top similar users for each user of prediction group
result = []
for f in range(len(samp_user)):
    sim = cosine_similarity(samp_user[f:f+1],matrix)
    top_index = sim.argsort()[0][-5:][::-1]
    #print(sim)
    #print(top_index)
    user_predict = (matrix[top_index[0]] + matrix[top_index[1]] + matrix[top_index[2]] + matrix[top_index[3]] + matrix[top_index[4]])/5 
    result.append(user_predict)
    #print(user_predict)
    index = user_predict.argmax(axis=0)
    print("The hero recommended for user %d is %d " % (f,index))

result_array = np.array(result)
from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, actual_value):
    return sqrt(mean_squared_error(prediction, actual_value))

print ('User-based CF RMSE: ' + str(rmse(result_array, samp_user)))







