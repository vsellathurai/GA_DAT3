# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 16:04:01 2017

@author: VickSella
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import statsmodels.formula.api as smf

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# read data into a DataFrame
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14
            
lm = smf.ols(formula='Sales ~ TV', data=data).fit()

data['TV_dollars'] = data.TV * 1000
feature_cols = ['TV']            


X = data[feature_cols]
y = data.Sales

linreg = LinearRegression()
linreg.fit(X,y)

lm.conf_int()


    