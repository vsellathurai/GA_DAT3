# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 15:24:19 2017

@author: UOB
"""

# glass identification dataset
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
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

plt.scatter(X,assorted_pred)
plt.plot(X, assorted_pred_prob, color='red')