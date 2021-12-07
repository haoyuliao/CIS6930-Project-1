# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import RegressorChain
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score,confusion_matrix
from joblib import dump, load
from sklearn.model_selection import KFold
import scipy.linalg as la
import warnings, os

warnings.filterwarnings("ignore")

###Predefined function.###
def value2binary(vec): #Turn numericdata into binary data.
    return [1 if x >= 0.5 else 0 for x in vec]

def make_custom_scorer(estimator, X_test, y_test): #Customer's accuracy.
    '''
    estimator: scikit-learn estimator, fitted on train data
    X_test: array-like, shape = [n_samples, n_features] Data for prediction
    y_test: array-like, shape = [n_samples] Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples] Predicted scores
    
    '''

    all_params = estimator.get_params() #Get currnet CV's trained parameters.
    y_score = estimator.predict(X_test) #Predict based on input X.
    for i in range(len(y_score)): #Turn the numeric  data into binary data.
        vec = y_score[i]
        y_score[i] = value2binary(vec)
    acc = accuracy_score(y_test, y_score) #Calculating ACC.    
    return acc

def CVforLinearRegression(X,y,thresholds): #Calculating optimal weigts for linear regression based the normal equation.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Divded data into training and testing.
    kf = KFold(n_splits=10) #K fold = 10.
    kf.get_n_splits(X_train)
    meanACCval=[]
    Weights = {}
    for t in thresholds: #The tresholds for truning numericdata into binary data
        ACCtrain = []
        ACCval = []
        for train_index, val_index in kf.split(X_train): #Strating CV process.
            X_ktrain, X_kval = X_train[train_index], X_train[val_index]
            y_ktrain, y_kval = y_train[train_index], y_train[val_index]
            X = X_ktrain
            PredTrainY = []
            PredTestY = []
            Weights[t]=[]
            Weigtmp = []
            for i in range(9): #Calculating weights for each column y.
                Y = y_ktrain[:,i]
                W = la.inv(X.T @ X) @ (X.T @ Y) #Calculating weights by normal equation.
                Weigtmp.append(W)
                PredTrainYk = X@W #Predict training Y.
                PredTrainYk = [1 if x >= t else 0 for x in PredTrainYk]
                PredTrainY.append(PredTrainYk) #Save predicted results.
                PredTestYk = X_kval@W #Predict testing Y.
                PredTestYk = [1 if x >= t else 0 for x in PredTestYk]
                PredTestY.append(PredTestYk) #Save predicted results.
            Weigtmp = np.array(Weigtmp) 
            Weights[t].append(Weigtmp) #Save weights with different thresholds.
            PredTrainY = np.array(PredTrainY).T
            PredTestY = np.array(PredTestY).T
            ACCval.append(accuracy_score(PredTestY, y_kval)) #Calculating each K fold overall accuracy.
        meanACCval.append(np.sum(ACCval)/len(ACCval)) #Calculating the mean of traning accuracy for each thresholds.

    meanACCval = np.array(meanACCval) #Find the bestest thresholds.
    bestThresholds = thresholds[np.argmax(meanACCval)]
    bestWeights = Weights[bestThresholds][0]

    #Showing the bestest weights of training accuracy and confusion matrix.
    y_trainPred = X_train @ bestWeights.T
    y_trainPred[y_trainPred >= bestThresholds] = 1
    y_trainPred[y_trainPred < bestThresholds] = 0
    print("Results of RegLN_multilabel model")
    ACCtrain = accuracy_score(y_train, y_trainPred).round(2)
    print("Accuracy for training: %s" %(ACCtrain))
    print('Confusion matrix for training:')
    for i in range(9):
        print("Confusion matrix for label {}:".format('x'+str(i)))
        print(confusion_matrix(y_train[:,i], y_trainPred[:,i],normalize='true').round(2))
    
    #Showing the bestest weights of testing accuracy and confusion matrix.
    y_testPred = X_test @ bestWeights.T
    y_testPred[y_testPred >= bestThresholds] = 1
    y_testPred[y_testPred < bestThresholds] = 0
    ACCtest = accuracy_score(y_test, y_testPred).round(2)
    print("Accuracy for testing: %s" %(ACCtest).round(2))
    print('Confusion matrix for testing:')
    for i in range(9):
        print("Confusion matrix for label {}:".format('x'+str(i)))
        print(confusion_matrix(y_test[:,i], y_testPred[:,i],normalize='true').round(2))

    #Save bestest trained parameters.
    np.save('./TrainedParameters/RegLN_multilabel',bestWeights)
    np.save('./TrainedParameters/RegLN_multilabel_Thresholds',bestThresholds)

    return ACCtrain, ACCtest
##########

###Load the dataset.###
MultiDataset = np.loadtxt('tictac_multi.txt')
MultiX = MultiDataset[:,:9]
Multiy = MultiDataset[:,9:]

#Shuffle data. 80% data for CV and 20% data for testing.
X_trainMulti, X_testMulti, y_trainMulti, y_testMulti = train_test_split(MultiX, Multiy, test_size=0.2, random_state=42)

#Current all model's name.
modelsName = ['RegKNN_multilabel',
              'RegLN_multilabel',
              'RegMLP_multilabel']

#Set up all models.
models = [RegressorChain(KNeighborsRegressor()),
          'RegLN_multilabel',
          RegressorChain(MLPRegressor())]

#Set up all models' parameters. The current parameters are the best.
modelsParameters =[{'base_estimator__n_neighbors': [22], #Fro KNN with multi lable.
              'base_estimator__weights':['distance'],
              'base_estimator__p':[1]},
                   {'thresholds':[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, #Fro LN with multi lable.
                                  0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]}, 
                   {'base_estimator__learning_rate': ["constant"], #Fro MLP with multi lable.
              'base_estimator__hidden_layer_sizes': [(500,20)],
              'base_estimator__alpha': [0.0001], # minimal effect
              'base_estimator__warm_start': [False], # minimal effect
              'base_estimator__momentum': [0.1], # minimal effect
              'base_estimator__learning_rate_init': [1e-3],
              'base_estimator__max_iter': [500],
              'base_estimator__random_state': [42],
              'base_estimator__solver':['adam'],
              'base_estimator__activation': ['relu']}]

#Save each CV's training accuracy and testing accuracy.
accTrainModels = [] 
accTestModels = []
scoring = {'cs1':make_custom_scorer}
for m in range(len(models)):
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('Start cross validation for %s model' %(modelsName[m]))
    #Cross validation with 10 k folds for linear regression.
    if modelsName[m] == 'RegLN_multilabel':
        ACCtrain,ACCtest = CVforLinearRegression(MultiX,Multiy,modelsParameters[m]['thresholds'])
        accTrainModels.append(ACCtrain)
        accTestModels.append(ACCtest)
        print("ʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌ")
        continue
    
    label = modelsName[m].split('_')[-1] 
    TrainValiRes = {'TrainAccuracy':[],'ValiAccuracy':[]}
    #Cross validation with 10 k folds.
    reg = GridSearchCV(models[m], param_grid=modelsParameters[m], scoring=scoring, cv=10, refit='cs1',return_train_score=True)
    reg.fit(X_trainMulti, y_trainMulti)

    idx = 0 #Saving each CV's results including parameters, train ACC, and Valdation ACC.
    for param in reg.cv_results_['params']:
        for k in param:
            keyName = k.split('__')[-1] #Get the name of parameter.
            if keyName not in TrainValiRes:
                TrainValiRes[keyName]=[]
            TrainValiRes[keyName].append(param[k])    
        TrainValiRes['TrainAccuracy'].append(reg.cv_results_['mean_train_cs1'][idx])
        TrainValiRes['ValiAccuracy'].append(reg.cv_results_['mean_test_cs1'][idx])
        idx+=1
    if not os.path.exists('TrainedCVresults'):
        os.makedirs('TrainedCVresults')
    TrainValiResToPd = pd.DataFrame(data=TrainValiRes)
    TrainValiResToPd.to_excel('./TrainedCVresults/'+modelsName[m]+'_Results.xlsx', index=True)

    print("Results of %s model" %(modelsName[m]))

    #The confusion matrix of multilabel will be 2 by 2 matrix for each column y.        
    #Show training results of accuracy and confusion matrix.                            
    y_trainPred = reg.predict(X_trainMulti)
    y_trainPred[y_trainPred >= 0.5] = 1
    y_trainPred[y_trainPred < 0.5] = 0
    ACCtrain = accuracy_score(y_trainMulti, y_trainPred).round(2)
    print("Accuracy for training: %s" %(ACCtrain))
    print('Confusion matrix for training:')
    for i in range(9):
        print("Confusion matrix for label {}:".format('x'+str(i)))
        print(confusion_matrix(y_trainMulti[:,i], y_trainPred[:,i],normalize='true').round(2))

    #Show testing results of accuracy and confusion matrix.
    y_testPred = reg.predict(X_testMulti)
    y_testPred[y_testPred >= 0.5] = 1
    y_testPred[y_testPred < 0.5] = 0
    ACCtest = accuracy_score(y_testMulti, y_testPred).round(2)
    print("Accuracy for testing: %s" %(ACCtest))
    print('Confusion matrix for testing:')
    for i in range(9):
        print("Confusion matrix for label {}:".format('x'+str(i)))
        print(confusion_matrix(y_testMulti[:,i], y_testPred[:,i],normalize='true').round(2))
    print("ʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌ")

    accTrainModels.append(ACCtrain)
    accTestModels.append(ACCtest)

    #Save trained model's parameters.
    if not os.path.exists('TrainedParameters'):
        os.makedirs('TrainedParameters')
    dump(reg, './TrainedParameters/'+modelsName[m]+'.joblib')

#Show bestest model's results.
accTrainModels = np.array(accTrainModels)
accTestModels = np.array(accTestModels)
print('Best classifier model is %s.' %(modelsName[np.argmax(accTestModels)]))
print('Best classifier model training accuracy %s.' %(accTrainModels[np.argmax(accTestModels)]))
print('Best classifier model testing accuracy %s.' %(accTestModels[np.argmax(accTestModels)]))

#Save all models' training and testing results.
totalRes = {}
totalRes['Model'] = modelsName
totalRes['Train ACC'] = accTrainModels
totalRes['Test ACC'] = accTestModels
totalResToPd = pd.DataFrame(data=totalRes)
totalResToPd.to_excel('RegressorsRes4AllModels.xlsx', index=True)
