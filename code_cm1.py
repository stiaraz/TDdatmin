# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:09:30 2019

@author: SYAVIRA TIARA Z
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils import resample
from imblearn.combine import SMOTETomek
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

dataset= pd.read_csv('dataset/cm1.csv', skiprows = range(0,354), header = None)

kelas= dataset[21]
print('Before: Class{}'. format(Counter(kelas)))
df = dataset.copy() #train itu yg X tanpa class
del df[21]

x_train, x_test, y_train, y_test = train_test_split(df, kelas, test_size= .1, random_state=10)

#PCA
#from sklearn.preprocessing import StandardScaler
#standardizedData = StandardScaler().fit_transform(df_res)
#pca = PCA(n_components=2)
#df_res = pca.fit_transform(X = standardizedData)

#balancing data
sm = SMOTETomek()
df_resm, kelas_res = sm.fit_sample(df, kelas)
#df_res_vis = pca.transform(df_resm)
print('After: Class{}'. format(Counter(kelas_res)))
##plotting imbalance
besar= dataset.groupby(kelas).size()
besar=list(besar)
koor_x=['false', 'true']
koor_y= besar

kelas_res= list(kelas_res)

valp= kelas_res.count("false")
valn= kelas_res.count("true")
new_y=[]
new_y.append(valp)
new_y.append(valn)
plt.bar(koor_x, new_y, label= 'After SMOTE+TOMEK', color='b', width=0.3, align='center')
plt.bar(koor_x, koor_y, label= 'Before SMOTE+TOMEK', color='r', width=0.3, align='edge')

plt.xlabel('class')
plt.ylabel('value')
plt.legend()
plt.show()

embedding = LocallyLinearEmbedding(n_components=5, method='ltsa')
# method='hessian', eigen_solver='dense'
X_transformed = embedding.fit_transform(df_resm)