# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:31:24 2017

@author: VickSella
"""
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

kf = KFold(X.shape[0], n_folds=5,random_state=1, shuffle=False)

#print '{} {:^61} {}'.format('Iteration', 'Training set observations', 'Testing set observations')
#for iteration, data in enumerate(kf, start=1):
#    print '{:^9} {} {:^25}'.format(iteration, data[0], data[1])
    
# TODO (super fun) - search for an optimal value of K for KNN
k_range = range(1, 31)
knn = KNeighborsClassifier(n_neighbors=k_range)
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')
k_scores = [cross_val_score(knn, X, y, cv=10, scoring='accuracy')]
# fill up k_scores!