# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:13:48 2019

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

dataset= pd.read_csv('dataset/kc1.csv', skiprows = range(0,356), header = None)

kelas= dataset[21]
print('Before: Class{}'. format(Counter(kelas)))
df = dataset.copy()
del df[21]

x_train, x_test, y_train, y_test = train_test_split(df, kelas, test_size= .1, random_state=10)

#balancing data
#sm = SMOTETomek()
#sm = SMOTE(random_state=42)
from imblearn.over_sampling import BorderlineSMOTE
sm = BorderlineSMOTE(random_state=42)
df_resm, kelas_res = sm.fit_sample(df, kelas)
from imblearn.under_sampling import TomekLinks
tl = TomekLinks()
df_resm2, kelas_res2 = tl.fit_sample(df_resm, kelas_res)
print('After: Class{}'. format(Counter(kelas_res)))
#df_res_vis = pca.transform(df_resm)


besar= dataset.groupby(kelas).size()
besar=list(besar)
koor_x=['false', 'true']
koor_y= besar
kelas_res= list(kelas_res)
valp= kelas_res.count(False)
valn= kelas_res.count(True)
new_y=[]
new_y.append(valp)
new_y.append(valn)
plt.bar(koor_x, new_y, label= 'After SMOTE+TOMEK', color='b', width=0.3, align='center')
plt.bar(koor_x, koor_y, label= 'Before SMOTE+TOMEK', color='r', width=0.3, align='edge')
plt.xlabel('class')
plt.ylabel('value')
plt.legend()
plt.show()

#normalization

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2=scaler.fit_transform(df_resm2)

#ltsa
embedding = LocallyLinearEmbedding(n_components=5, method='ltsa', eigen_solver='dense')
X_transformed = embedding.fit_transform(df_resm)

#cfs
df2=pd.DataFrame(df2)

import seaborn as sns
import numpy as np
corr = df2.corr()

sns.heatmap(corr)

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df2.columns[columns]
df2= df2[selected_columns]

#svm and grid search
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
parameters = {'C':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]}
svm = SVC(kernel='rbf',gamma=10)
clf = GridSearchCV(svm, parameters, cv=10)
clf.fit(df_resm, kelas_res)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print(clf.best_score_)

bestmodel = clf.best_estimator_

from sklearn.model_selection import cross_validate
scoring = ['accuracy','precision_macro', 'recall_macro' ]
scores = cross_validate(bestmodel, df_resm, kelas_res, cv=10, scoring =scoring)

print("Accuracy: %0.4f " % (scores['test_accuracy'].mean()))
print("RECALL: %0.4f " % (scores['test_precision_macro'].mean()))
print("PRESISI: %0.4f )" % (scores['test_recall_macro'].mean()))

y_pred = bestmodel.predict(df_resm)

from sklearn.metrics import classification_report
y_true_svm = kelas_res
y_pred_svm = y_pred
print(classification_report(y_true_svm, y_pred_svm))


from sklearn import metrics
csvmtrain = metrics.confusion_matrix(y_true_svm, y_pred)
print(csvmtrain)