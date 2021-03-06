#!/usr/bin/env python
# coding: utf-8
#Program for 'OR' Black Box Detector

import pandas as pd
import pickle
import numpy 
import numpy as np
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPooling1D
from numpy import unique
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def predict(genVector, selectAlgo):
    #svm
    if selectAlgo == 'SVM':
        filenameSvm = 'blackboxmodel/blackbox_svm_model.pickle'
        loaded_modelSvm = pickle.load(open(filenameSvm, 'rb'))
        result = loaded_modelSvm.predict(genVector)

    #rf Model
    if selectAlgo == 'RF':
        filenameRf = 'blackboxmodel/blackbox_rf_model.pickle'
        loaded_modelRf = pickle.load(open(filenameRf, 'rb'))
        result = loaded_modelRf.predict(genVector)

    #xgBoost Model
    if selectAlgo == 'XgBoost':
        filenameXgBoost= 'blackboxmodel/blackbox_xgBoost_model.pickle'
        loaded_modelXgBoost = pickle.load(open(filenameXgBoost, 'rb'))
        result = loaded_modelXgBoost.predict(genVector)

    #dt Model
    if selectAlgo == 'DT':
        filenameDt = 'blackboxmodel/blackbox_dt_model.pickle'
        loaded_modelDt = pickle.load(open(filenameDt, 'rb'))
        result = loaded_modelDt.predict(genVector)

    #lr Model
    if selectAlgo == 'LR':
        filenameLr = 'blackboxmodel/blackbox_lr_model.pickle'
        loaded_modelLr = pickle.load(open(filenameLr, 'rb'))
        result = loaded_modelLr.predict(genVector)
    
    #1D CNN Model
    if selectAlgo == 'CNN':
        genVector_cnn = genVector.reshape(genVector.shape[0], genVector.shape[1], 1)
        loaded_modelCnn = keras.models.load_model('blackboxmodel/blackbox_1d_cnn.h5')
        resultCnn_ = loaded_modelCnn.predict(genVector_cnn)
        result = np.argmax(resultCnn_ , axis=1)

    #result = [(resultSvm[i] or resultRf[i] or resultXgBoost[i] or resultLr[i] or resultDt[i] or resultCnn[i] ) for i in range(len(resultSvm))]
    #algoResult = {'SVM': numpy.array(resultSvm), 'RF': numpy.array(resultRf), 'XgBoost': numpy.array(resultXgBoost), 'LR': numpy.array(resultLr), 'DT': numpy.array(resultDt), 'CNN': numpy.array(resultCnn)}

    return numpy.array(result)


#score/accuracy calculation
def score(genVector, vectorLabel, selectAlgo):
    #loading model
    #svm
    if selectAlgo == 'SVM':
        filenameSvm = 'blackboxmodel/blackbox_svm_model.pickle'
        loaded_modelSvm = pickle.load(open(filenameSvm, 'rb'))
        result = loaded_modelSvm.predict(genVector)

    #rf Model
    if selectAlgo == 'RF':
        filenameRf = 'blackboxmodel/blackbox_rf_model.pickle'
        loaded_modelRf = pickle.load(open(filenameRf, 'rb'))
        result = loaded_modelRf.predict(genVector)

    #xgBoost Model
    if selectAlgo == 'XgBoost':
        filenameXgBoost= 'blackboxmodel/blackbox_xgBoost_model.pickle'
        loaded_modelXgBoost = pickle.load(open(filenameXgBoost, 'rb'))
        result = loaded_modelXgBoost.predict(genVector)

    #dt Model
    if selectAlgo == 'DT':
        filenameDt = 'blackboxmodel/blackbox_dt_model.pickle'
        loaded_modelDt = pickle.load(open(filenameDt, 'rb'))
        result = loaded_modelDt.predict(genVector)

    #lr Model
    if selectAlgo == 'LR':
        filenameLr = 'blackboxmodel/blackbox_lr_model.pickle'
        loaded_modelLr = pickle.load(open(filenameLr, 'rb'))
        result = loaded_modelLr.predict(genVector)
    
    #1D CNN Model
    if selectAlgo == 'CNN':
        genVector_cnn = genVector.reshape(genVector.shape[0], genVector.shape[1], 1)
        loaded_modelCnn = keras.models.load_model('blackboxmodel/blackbox_1d_cnn.h5')
        resultCnn_ = loaded_modelCnn.predict(genVector_cnn)
        result = np.argmax(resultCnn_ , axis=1)

    #result = [(resultSvm[i] or resultRf[i] or resultXgBoost[i] or resultLr[i] or resultDt[i] or resultCnn[i])  for i in range(len(resultSvm))]
    #result = numpy.array(result)
    #TPR = accuracy_score(result, vectorLabel)

    #algoResult = {'SVM': resultSvm, 'RF': resultRf, 'XgBoost': resultXgBoost, 'Lr': resultLr, 'Dt': resultDt, 'CNN': resultCnn}

    algoDF = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])

    #for algo, result in algoResult.items():
    accuracy = accuracy_score(vectorLabel, result, sample_weight=None)
    precision = precision_score(vectorLabel, result)
    recall = recall_score(vectorLabel, result)
    f1 = f1_score(vectorLabel, result)
    algoDF.loc[0] = [accuracy, precision, recall, f1]
    algoDF.to_csv(f'./{selectAlgo}.csv', mode='a', index=True, header=True)

    return accuracy_score(vectorLabel, result, sample_weight=None)
    
#train
def fit(X,y, algo):
    #cnn
    if algo == 'CNN':
        X_ = X.reshape(X.shape[0], X.shape[1], 1)
        modelCnn = Sequential()
        modelCnn.add(Conv1D(64, 2, activation="relu", input_shape=(X.shape[1],1)))
        modelCnn.add(Dense(16, activation="relu"))
        modelCnn.add(MaxPooling1D())
        modelCnn.add(Flatten())
        modelCnn.add(Dense(2, activation = 'softmax'))
        modelCnn.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = ['accuracy'])
        modelCnn.fit(X_, y, batch_size=64, epochs=15, verbose=0)
        modelCnn.save("blackboxmodel/blackbox_1d_cnn.h5")

    #xtrain = DataLoader(X, batch_size=64)
    #ytrain = DataLoader(y, batch_size=64)
    """classifierSvm = SVC(gamma='scale',kernel='linear',probability=True)
    classifierRf = RandomForestClassifier(n_estimators = 300, criterion = 'gini') 
    classifierLr = LogisticRegression()
    classifierDt = DecisionTreeClassifier()
    classifierXgb = XGBClassifier(random_state=42,verbosity = 0)

    for ((idx, X), (idy, y)) in zip(enumerate(xtrain), enumerate(ytrain)):
        classifierSvm.fit(X, y) 
        classifierRf.fit(X, y)
        XgbX = np.array(X)
        XgbY = np.array(y)
        classifierXgb.fit(XgbX, XgbY)
        classifierLr.fit(X, y)
        classifierDt.fit(X, y)"""

    
    #svm
    if algo == 'SVM':
        classifierSvm = SVC(gamma='scale',kernel='linear', probability=True)
        classifierSvm.fit(X, y)
        with open("blackboxmodel/blackbox_svm_model.pickle", "wb") as file_:
            pickle.dump(classifierSvm, file_, -1)
    
    #rf
    if algo == 'RF':
        classifierRf = RandomForestClassifier(n_estimators = 300, criterion = 'gini')
        classifierRf.fit(X, y)
        with open("blackboxmodel/blackbox_rf_model.pickle", "wb") as file_:
            pickle.dump(classifierRf, file_, -1)
    
    #xgb classifier
    if algo == 'XgBoost':
        classifierXgb = XGBClassifier(random_state=42, verbosity = 0)
        XgbX = np.array(X)
        XgbY = np.array(y)
        classifierXgb.fit(XgbX, XgbY)
        with open("blackboxmodel/blackbox_xgBoost_model.pickle", "wb") as file_:
            pickle.dump(classifierXgb, file_, -1)
    
    #lr
    if algo == 'LR':
        classifierLr = LogisticRegression()
        classifierLr.fit(X, y)
        with open("blackboxmodel/blackbox_lr_model.pickle", "wb") as file_:
            pickle.dump(classifierLr, file_, -1)
    
    #dt
    if algo == 'DT':
        classifierDt = DecisionTreeClassifier()
        classifierDt.fit(X, y)
        with open("blackboxmodel/blackbox_dt_model.pickle", "wb") as file_:
            pickle.dump(classifierDt, file_, -1)
        

