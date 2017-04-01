# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 14:19:53 2017

@author: UOB
"""

from sklearn.datasets import make_blobs

num_blobs = 8
X, Y = make_blobs(centers=num_blobs, cluster_std=0.5, random_state=2)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans



kmeans = KMeans(n_clusters=8, random_state=0).fit(X)
kmeans.labels_
kmeans.predict

plt.scatter(X[:,0],X[:,1], c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],X[:,1], c=kmeans.labels_)