# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 15:21:29 2017

@author: UOB
"""

# beer dataset
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/beer.txt'
beer = pd.read_csv(url, sep=' ')

X = beer.drop('name',axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

km = KMeans(n_clusters=3,random_state=0).fit(X)
beer['cluster'] = km.labels_
beer.sort_values(by='cluster')

centers = beer.groupby('cluster').mean()

import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14
            
colors = np.array(['red', 'green', 'blue', 'yellow'])

## scatter plot of calories versus alcohol, colored by cluster (0=red, 1=green, 2=blue)
#plt.scatter(beer.calories, beer.alcohol, c=colors[beer.cluster], s=50)
#
## cluster centers, marked by "+"
#plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')
#
## add labels
#plt.xlabel('calories')
#plt.ylabel('alcohol')
#
#pd.scatter_matrix (X, c=colors[beer.cluster],figsize=(10,10), s=100)

k_range = (2,19)
sc = []
for k in k_range:
    kmm= KMeans(n_clusters=k,random_state=0).fit(X)
    scores = metrics.silhouette_score(X_scaled, kmm.labels_, metric='euclidean')
    sc.append(scores)
    
plt.plot(k_range,sc)