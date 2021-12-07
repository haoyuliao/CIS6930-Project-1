# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from joblib import dump, load
import warnings, os
from random import randint
import random

warnings.filterwarnings("ignore")

addNoise = False
redData = False
print('If variable of addNoise is True, the noise wiil be added.')
print('If variable of redData is True, the data will be decreased by 9/10.')

###Load the dataset.###
MultiDataset = np.loadtxt('tictac_multi.txt')
MultiX = MultiDataset[:,:9]
Multiy = MultiDataset[:,9:]

SingleDataset = np.loadtxt('tictac_single.txt')
SingleX = SingleDataset[:,:9]
Singley = SingleDataset[:,9:]

if addNoise:
    noise = np.random.normal(0, 1, Multiy.shape) #adding noise to truth labels.
    Multiy += noise
    noise = np.random.normal(0, 1, Singley.shape) #adding noise to truth labels.
    Singley += noise
    #After adding noise, turn numeral labels into binary labels.
    #Otherwise, the confusion matirx and accuracy cannot be calculated.
    Singley[Singley >= 0.5] = 1 #If the number >= 0.5, turn into 1.
    Singley[Singley < 0.5] = 0
    Multiy[Multiy >= 0.5] = 1
    Multiy[Multiy < 0.5] = 0
if redData:
    rN = int(len(MultiDataset)-len(MultiDataset)*9/10) #decrease data by a 9 of 10.
    MultiX = MultiX[0:rN]
    Multiy = Multiy[0:rN]
    SingleX = SingleX[0:rN]
    Singley = Singley[0:rN]

#Shuffle data. 80% data for CV and 20% data for testing.
X_trainMulti, X_testMulti, y_trainMulti, y_testMulti = train_test_split(MultiX, Multiy, test_size=0.2, random_state=42)
X_trainSingle, X_testSingle, y_trainSingle, y_testSingle = train_test_split(SingleX, Singley, test_size=0.2, random_state=42)

#Current all model's name.
modelsName = ['ClfLinearSVM_multilabel',
              'ClfRBFSVM_multilabel',
              'ClfKNN_multilabel',
              'ClfMLP_multilabel',
              'ClfLinearSVM_singlelabel',
              'ClfRBFSVM_singlelabel',
              'ClfKNN_singlelabel',
              'ClfMLP_singlelabel']
#Set up all models.
models = [MultiOutputClassifier(SVC(kernel='linear')),
          MultiOutputClassifier(SVC(kernel='rbf')),
          MultiOutputClassifier(KNeighborsClassifier()),
          MLPClassifier(),
          SVC(kernel='linear'),
          SVC(kernel='rbf'),
          KNeighborsClassifier(),
          MLPClassifier()]

#Set up all models' parameters. The current parameters are the best.
modelsParameters =[{'estimator__gamma':['scale'], #For linear SVM with multi lable.
              'estimator__coef0':[0],
              'estimator__tol':[1e-2]
              ,'estimator__C':[1],
              'estimator__max_iter':[-1]},
                   {'estimator__gamma':['scale'], #For RBF SVM with multi lable.
              'estimator__coef0':[0],
              'estimator__tol':[1e-2]
              ,'estimator__C':[900],
              'estimator__max_iter':[-1]},
                   {'estimator__n_neighbors': [35], #For KNN with multi lable.
              'estimator__weights':['distance'],
              'estimator__p':[1]},
                   {'learning_rate': ["constant"], #Fro MLP with multi lable.
              'hidden_layer_sizes': [(500,20)],
              'alpha': [0.0001], # minimal effect
              'warm_start': [False], # minimal effect
              'momentum': [0.1], # minimal effect
              'learning_rate_init': [1e-3],
              'max_iter': [800],
              'random_state': [42],
              'solver':['adam'],
              'activation': ['relu']},
                   {'gamma':['scale'], #For linear SVM with singel lable.
              'coef0':[0],
              'tol':[1e-2],
              'C':[1],
              'max_iter':[-1]},
                   {'gamma':['scale'], #For RBF SVM with singel lable.
              'coef0':[0],
              'tol':[1e-2],
              'C':[100],
              'max_iter':[-1]},
                   {'n_neighbors': [38], #For KNN with singel lable.
              'weights':['distance'],
              'p':[1]},
                   {'learning_rate': ["constant"], #Fro MLP with singel lable.
              'hidden_layer_sizes': [(500,20)],
              'alpha': [0.0001], # minimal effect
              'warm_start': [False], # minimal effect
              'momentum': [0.1], # minimal effect
              'learning_rate_init': [1e-3],
              'max_iter': [500],
              'random_state': [42],
              'solver':['adam'],
              'activation': ['relu']}]

#Save each CV's training accuracy and testing accuracy.
accTrainModels = [] 
accTestModels = [] 
for m in range(len(models)):
    label = modelsName[m].split('_')[-1]
    TrainValiRes = {'TrainAccuracy':[],'ValiAccuracy':[]}
    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
    print('Start cross validation for %s model' %(modelsName[m]))
    #Cross validation with 10 k folds.
    clf = GridSearchCV(models[m], param_grid=modelsParameters[m], scoring={'ACC': 'accuracy'}, cv=10, refit='ACC',return_train_score=True)
    if label == 'multilabel': #Training model.
        clf.fit(X_trainMulti, y_trainMulti)
    if label == 'singlelabel':
        clf.fit(X_trainSingle, y_trainSingle)
        
    idx = 0 #Saving each CV's results including parameters, train ACC, and Valdation ACC.
    for param in clf.cv_results_['params']:
        for k in param:
            keyName = k.split('__')[-1] #Get the name of parameter.
            if keyName not in TrainValiRes:
                TrainValiRes[keyName]=[]
            TrainValiRes[keyName].append(param[k])    
        TrainValiRes['TrainAccuracy'].append(clf.cv_results_['mean_train_ACC'][idx])
        TrainValiRes['ValiAccuracy'].append(clf.cv_results_['mean_test_ACC'][idx])
        idx+=1
    if not os.path.exists('TrainedCVresults'):
        os.makedirs('TrainedCVresults')
    TrainValiResToPd = pd.DataFrame(data=TrainValiRes)
    TrainValiResToPd.to_excel('./TrainedCVresults/'+modelsName[m]+'_Results.xlsx', index=True)

    print("Results of %s model" %(modelsName[m]))
    if label == 'multilabel':
        #The confusion matrix of multilabel will be 2 by 2 matrix for each column y.        
        #Show training results of accuracy and confusion matrix.              
        y_trainPred = clf.predict(X_trainMulti)
        ACCtrain = accuracy_score(y_trainMulti, y_trainPred).round(2)
        print("Accuracy for training: %s" %(ACCtrain))
        print('Confusion matrix for training:')
        for i in range(9):
            print("Confusion matrix for label {}:".format('x'+str(i)))
            print(confusion_matrix(y_trainMulti[:,i], y_trainPred[:,i],normalize='true').round(2))

        #Show testing results of accuracy and confusion matrix.
        y_testPred = clf.predict(X_testMulti)
        ACCtest = accuracy_score(y_testMulti, y_testPred).round(2)
        print("Accuracy for testing: %s" %(ACCtest))
        print('Confusion matrix for testing:')
        for i in range(9):
            print("Confusion matrix for label {}:".format('x'+str(i)))
            print(confusion_matrix(y_testMulti[:,i], y_testPred[:,i],normalize='true').round(2))

    if label == 'singlelabel':
        #Show training results of accuracy and confusion matrix.   
        y_trainPred = clf.predict(X_trainSingle)
        ACCtrain = accuracy_score(y_trainSingle, y_trainPred).round(2)
        print("Accuracy for training: %s" %(ACCtrain))
        print('Confusion matrix for training:')
        print(confusion_matrix(y_trainSingle, y_trainPred,normalize='true').round(2))

        #Show testing results of accuracy and confusion matrix.
        y_testPred = clf.predict(X_testSingle)
        ACCtest = accuracy_score(y_testSingle, y_testPred).round(2)
        print("Accuracy for testing: %s" %(ACCtest))
        print('Confusion matrix for testing:')
        print(confusion_matrix(y_testSingle, y_testPred,normalize='true').round(2))
    print("ʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌʌ")

    accTrainModels.append(ACCtrain)
    accTestModels.append(ACCtest)

    #Save trained model's parameters.
    if not os.path.exists('TrainedParameters'):
        os.makedirs('TrainedParameters')
    dump(clf, './TrainedParameters/'+modelsName[m]+'.joblib')

#Show bestest model's results.
accTrainModels = np.array(accTrainModels)
accTestModels = np.array(accTestModels)
print('Best classifier model is %s.' %(modelsName[np.argmax(accTestModels)]))
print('Best classifier model training accuracy %s.' %(accTrainModels[np.argmax(accTestModels)]))
print('Best classifier model testing accuracy %s.' %(accTestModels[np.argmax(accTestModels)]))

#Save all models' training and testing records.
totalRes = {}
totalRes['Model'] = modelsName
totalRes['Train ACC'] = accTrainModels
totalRes['Test ACC'] = accTestModels
totalResToPd = pd.DataFrame(data=totalRes)
totalResToPd.to_excel('ClassifiersRes4AllModels.xlsx', index=True)
##############################################################################