# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:26:57 2019

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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding

dts= pd.read_csv('dataset/pc3.csv', skiprows = range(0,45), header = None)
dts=pd.DataFrame(dts.replace('?', 0))
#dts=dataset.fillna(0)
dataset = dts.groupby(dts.columns, axis = 1).transform(lambda x: x.fillna(x.mean()))
kelas= dataset[40]
print('Before: Class{}'. format(Counter(kelas)))
df = dataset.copy()
del df[40]

#x_train, x_test, y_train, y_test = train_test_split(df, kelas, test_size= .1, random_state=10)

#PCA
#from sklearn.preprocessing import StandardScaler
#standardizedData = StandardScaler().fit_transform(df_res)
#pca = PCA(n_components=2)
#df_res = pca.fit_transform(X = standardizedData)

sm = SMOTETomek()
#sm = SMOTE(random_state=42)
df_resm, kelas_res = sm.fit_sample(df, kelas)
print('After: Class{}'. format(Counter(kelas_res)))

besar= dataset.groupby(kelas).size()
besar=list(besar)
koor_x=['false', 'true']
koor_y= besar
kelas_res= list(kelas_res)
valp= kelas_res.count("N")
valn= kelas_res.count("Y")
new_y=[]
new_y.append(valp)
new_y.append(valn)
plt.bar(koor_x, new_y, label= 'After SMOTE+TOMEK', color='b', width=0.3, align='center')
plt.bar(koor_x, koor_y, label= 'Before SMOTE+TOMEK', color='r', width=0.3, align='edge')
plt.xlabel('class')
plt.ylabel('value')
plt.legend()
plt.show()

embedding = LocallyLinearEmbedding(n_components=5, method='ltsa',eigen_solver='dense')
# method='hessian', eigen_solver='dense'
X_transformed = embedding.fit_transform(df_resm)