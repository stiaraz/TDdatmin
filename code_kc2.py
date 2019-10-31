# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:00:06 2019

@author: SYAVIRA TIARA Z
@co-author : ANA AZ
"""

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from imblearn.combine import SMOTETomek
from sklearn.manifold import LocallyLinearEmbedding
from imblearn.over_sampling import SMOTE
dataset= pd.read_csv('dataset/kc2.csv', skiprows = range(0,359), header = None)

kelas= dataset[21]
print('Before: Class{}'. format(Counter(kelas)))
df = dataset.copy() #train itu yg X tanpa class
del df[21]

#normalization

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df=scaler.fit_transform(df)

#oversampling

sm = SMOTETomek()
df_resm, kelas_res = sm.fit_sample(df, kelas)

print('After: Class{}'. format(Counter(kelas_res)))
besar= dataset.groupby(kelas).size()
besar=list(besar)
koor_x=['no', 'yes']
koor_y= besar
kelas_res= list(kelas_res)
valp= kelas_res.count("no")
valn= kelas_res.count("yes")
new_y=[]
new_y.append(valp)
new_y.append(valn)
plt.bar(koor_x, new_y, label= 'After SMOTE+TOMEK', color='b', width=0.3, align='center')
plt.bar(koor_x, koor_y, label= 'Before SMOTE+TOMEK', color='r', width=0.3, align='edge')

plt.xlabel('class')
plt.ylabel('value')
plt.legend()
plt.show()
#ltsa

embedding = LocallyLinearEmbedding(method='ltsa',eigen_solver='dense', n_components=5)
X_transformed = embedding.fit_transform(df_resm)

#correlation

X_transformed=pd.DataFrame(X_transformed)

import seaborn as sns
import numpy as np
corr = X_transformed.corr()

sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = X_transformed.columns[columns]
X_transformed = X_transformed[selected_columns]

#svm and grid search

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'C':[1, 10]}
svm = SVC(kernel='rbf',gamma='auto')
clf = GridSearchCV(svm, parameters, cv=10)
clf.fit(X_transformed, kelas_res)
print(clf.best_score_)
