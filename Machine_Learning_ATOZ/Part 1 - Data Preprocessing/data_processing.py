#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 11:14:43 2019

@author: rahul
"""

# data pre processing
#Importing the Libraries

# Mathematical tools
import numpy as np
# Plot charts
import matplotlib.pyplot as plt
# import and manage datasets
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Missing Data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "mean", missing_values=np.nan)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding Categroical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([("Transform Countries", OneHotEncoder(), [0])], remainder = "passthrough")
X = ct.fit_transform(X)


labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(Y)


# Preparing training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


