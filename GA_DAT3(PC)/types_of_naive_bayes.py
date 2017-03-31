# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:48:31 2017

@author: VickSella
"""

# read the data
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv(url, header=None, names=col_names)
pima= pima.drop('label',1)

feature_cols = ['pregnant','bp','skin','insulin','bmi','pedigree','age']
X= pima[feature_cols]
y= pima.glucose

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)

# testing accuracy of Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
y_pred_class = mnb.predict(X_test)

print accuracy_score(y_test,y_pred_class)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_class_g = gnb.predict(X_test)

print accuracy_score(y_test,y_pred_class_g)