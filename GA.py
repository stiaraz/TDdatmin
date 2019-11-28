# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 14:40:13 2019

@author: SYAVIRA TIARA Z
"""
from sklearn import svm
import numpy as np
import pandas as pd
import random
import math
import operator
from sklearn.metrics import confusion_matrix
import sklearn.metrics as sm
from collections import Counter
from sklearn.model_selection import StratifiedKFold
""" Menentukan individu awal dalam populasi """
def kromosom(k):
    population=k
    return [random.randint(1,5) for x in range(population)]
    
""" Mengubah nilai individu menjadi bilangan biner """            
def binary(parent, length):
    biner=[]
    length="{0:0b}".format(length)
    length=len(length)
    for x in range(len(parent)):
        temp="{0:0b}".format(parent[x])
        biner.append(temp.zfill(length))
    return biner

""" Konversi biner ke desimal """    
def desimal(biner):
    return [int(x,2) for x in biner]

""" Memisahkan data latih dan data tes (opsional) """
def splitDataset(data, splitRatio):
    split=random.sample(data.index, (int)(splitRatio*len(data)))
    train= data.ix[split]
    test= data.drop(split)
    train.sort_index(inplace=True)
    test.sort_index(inplace=True)
    return [train, test]

""" Mencari nilai fitnes berdasarkan akurasi """
def fitness(x):
    # dilakukan svm
    n_fold = 10
    skf = StratifiedKFold(n_splits = n_fold)
    clf = svm.SVC(C= x, kernel='rbf',gamma = 100, tol=0.001, max_iter = -1)
    svm_fitness = 0
    for train_index, test_index in skf.split(dataset_minmax, class_data):
        attribute = []
        kelas = []
        for x in train_index:
            attribute.append(dataset_minmax[x])
            kelas.append(class_data[x])
#        x_train = pd.DataFrame(attribute)
#        y_train = pd.DataFrame(kelas)
        x_train = attribute
        y_train = kelas
        attribute = []
        kelas = []
        for y in test_index:
            attribute.append(dataset_minmax[y])
            kelas.append(class_data[y])
#        x_test = pd.DataFrame(attribute)
#        y_test = pd.DataFrame(kelas)
        x_test = attribute
        y_test = kelas
        svc = clf.fit(x_train, y_train)
        svm_result = svc.predict(x_test)
        correct = 0
#        svm_result = pd.DataFrame(knn_result)
#        print(str(len(y_test)) + " : " + str(len(svm_result)))
#        print type(y_test)
#        print type(knn_result)
        for i in range(len(y_test)):
#            print str(y_test[i])
#            print str(svm_result[i])
            if str(y_test[i]) == str(svm_result[i]):
                correct += 1
        fitness_value = correct / float(len(y_test))
        svm_fitness += fitness_value
        
    return svm_fitness/n_fold

""" Mencari nilai probabilitas kumulatif nilai fitnes yang akan digunakan untuk roulette wheel """
def cumulatived():
    fit=[]
    for x in Krom:
        if x == 0:
            continue
        else:
            fit.append(fitness(x))
    
    cumulative=[]
    prob=[]    
    for x in range(len(Krom)):
#        print(str(fit[x]) + ":" + str(sum(fit)))
        prob.append(fit[x]/sum(fit))
        if x!=0:
            cumulative.append(prob[x]+cumulative[x-1])        
        else:
            cumulative.append(prob[x])        
    return prob, cumulative

""" Seleksi roulette wheel """
def rouletteWheel():
    roulette=[]
    Ran=[]
    for x in range(len(cumulative)):
        R=random.random()    
        Ran.append(R)
#        print R
        for i in range(len(cumulative)):
            if R<cumulative[i]:
               roulette.append(Krom[i])    
#               print(str(i) + "=" + str(Krom[i]))
               break
    
    return roulette, Ran

""" Menghasilkan individu baru dengan metode crossover """
def crossover(length):
    R2=0
    parent=[]
    for x in range(int(k*0.8)):
        R1=random.randint(0,k-1)
        while R1==R2:
            R1=random.randint(0,k-1)
        parent.append(Krom[R1])
        R2=R1
    bins=binary(parent, length)

    child=[]    
    for x in range(len(parent)):
        if x%2==1:
            continue
        R1=random.randint(1,len(bins[x])-1)
        male=bins[x][:R1]
        female=bins[x][R1:]
        male2=bins[x+1][:R1]
        female2=bins[x+1][R1:]

        while desimal(male+female2)==0 or desimal(male2+female)==0:
            R1=random.randint(0,len(bins[x])-1)
            male=bins[x][:R1]
            female=bins[x][R1:]
            male2=bins[x+1][:R1]
            female2=bins[x+1][R1:]

        child.append(male+female2)
        child.append(male2+female)
        
    return child


if __name__ == '__main__':
    try :
#        dataset = pd.read_csv('blood.csv', skiprows = [0], header=None)
        dataset= pd.read_csv('dataset/kc1.csv', skiprows = range(0,356), header = None)

        classs= dataset[21]
        print('Before: Class{}'. format(Counter(classs)))
        df = dataset.copy()
        del df[21]
#        class_data = dataset[4]
#        del dataset[4]
        from imblearn.combine import SMOTETomek
        smt = SMOTETomek()
        df_resm, class_data = smt.fit_sample(df, classs)
        print('After: Class{}'. format(Counter(class_data)))
        #Normalisasi Data
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()
        dataset_minmax = min_max_scaler.fit_transform(df_resm)        
#        dataset_minmax=pd.DataFrame(dataset_minmax2)
#
#        import seaborn as sns
#        import numpy as np
#        corr = dataset_minmax.corr()
#        
#        sns.heatmap(corr)
#        
#        columns = np.full((corr.shape[0],), True, dtype=bool)
#        for i in range(corr.shape[0]):
#            for j in range(i+1, corr.shape[0]):
#                if corr.iloc[i,j] >= 0.9:
#                    if columns[j]:
#                        columns[j] = False
#        selected_columns = dataset_minmax.columns[columns]
#        dataset_minmax= dataset_minmax[selected_columns]
        k = 0
        Krom = []
        while k < 10:
            i = random.randint(1,20)
            if i == 0:
                continue
            elif i not in Krom:
                Krom.append(i)
                k += 1
            else:
                continue
        print (Krom)
        new_child=[]
        for x in range(1,31):
#            print "loop = "+str(x)
            Krom=Krom[:k] + new_child
            cumulatived()
            prob_fitness, cumulative=cumulatived()
            roulette, Random_log=rouletteWheel()        
            Krom=roulette
            cross=crossover(int(len(dataset)/10), )
            new_child=desimal(cross)
    
        k_fixed=prob_fitness.index(max(prob_fitness))
        print ("c optimal     : " + str(Krom[k_fixed]))
        n_fold = 10
        skf = StratifiedKFold(n_splits = n_fold )
        clf = svm.SVC(C= Krom[k_fixed], kernel='rbf',gamma = 100, tol=0.001, max_iter = -1)
        svm_accuracy = 0
        for train_index, test_index in skf.split(df, class_data):
            attribute = []
            kelas = []
            for x in train_index:
                attribute.append(dataset_minmax[x])
                kelas.append(class_data[x])
            x_train = pd.DataFrame(attribute)
            y_train = pd.DataFrame(kelas)
            attribute = []
            kelas = []
            for y in test_index:
                attribute.append(dataset_minmax[y])
                kelas.append(class_data[y])
            x_test = pd.DataFrame(attribute)
            y_test = pd.DataFrame(kelas)
            svc = clf.fit(x_train, y_train)
            svm_result = svc.predict(x_test)
            akurasi = float(sm.accuracy_score(svm_result, y_test)) * 100
            svm_accuracy += akurasi
#            print("\nAkurasi : " + str(akurasi) +"%")
            
        print('Akurasi Support Vector Machine : ' + repr(svm_accuracy / n_fold))

        
    except IOError:
        print('An error occured.')
