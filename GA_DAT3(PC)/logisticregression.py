# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 15:06:32 2017

@author: VickSella
"""

# glass identification dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass['assorted'] = glass.glass_type.map({1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1})

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

feature_cols = ['al']
X = glass[feature_cols]
y = glass.assorted

logreg = LogisticRegression(C=1e9)
logreg.fit(X,y)
assorted_pred = logreg.predict(X)
assorted_pred_prob = logreg.predict_proba(X)
assorted_pred_class = np.where(assorted_pred_prob <0.5, 1, 0)
glass['assorted_pred_class'] = zip(*assorted_pred_prob)[1]
glass.sort_values(by='al', inplace= True)
     
plt.scatter(glass.al,glass.assorted)
plt.plot(glass.al, glass.assorted_pred_class, color='red')

logodds = logreg.intercept_ + logreg.coef_*3
odds = np.exp(logodds)
prob = odds/(1+odds)
print prob