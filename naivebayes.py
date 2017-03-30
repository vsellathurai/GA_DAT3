# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:54:21 2017

@author: UOB
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 30)

critics = pd.read_csv('https://raw.githubusercontent.com/gfleetwood/fall-2014-lessons/master/datasets/rt_critics.csv')

text = ['Math is great', 'Math is really great', 'Exciting exciting Math']
print "Original text:\n\t", '\n\t'.join(text)

cv = CountVectorizer(ngram_range=(1, 2))

X = cv.fit_transform(critics.quote)
Y = critics.fresh
print x_back