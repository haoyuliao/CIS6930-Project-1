# -*- coding: utf-8 -*-
import numpy as np
from random import randint
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
import warnings, os
from joblib import dump

warnings.filterwarnings("ignore")

################GA-ClfMLP_singlelabel#############
#Credits from https://medium.com/analytics-vidhya/a-genetic-algorithm-for-optimizing-neural-network-parameters-d8187d5114ed
def inicializacao_populacao_mlp(size_mlp): #Initial population.
    activation = ['tanh', 'relu']
    solver = ['sgd', 'adam']
    pop =  np.array([[random.choice(activation), random.choice(solver), randint(2,100),randint(2,50)]])
    for i in range(0, size_mlp-1):
        pop = np.append(pop, [[random.choice(activation), random.choice(solver), randint(2,50),randint(2,50)]], axis=0)
    return pop

def cruzamento_mlp(pai_1, pai_2): #Crossover with two parents.
    child = []
    for i in range(5//2+5%2):
        if i*2 < len(pai_1):
            child.append(pai_1[i*2])
        if i*2+1 < len(pai_1):
            child.append(pai_2[i*2+1])  
    return child

def mutacao_mlp(child, prob_mut): #Mutation for child.
    child_ = np.copy(child)
    for c in range(0, len(child_)):
        if np.random.rand() >= prob_mut:
            k = randint(2,3)
            child_[c,k] = int(child_[c,k]) + randint(1, 4)
    return child_


def function_fitness_mlp(pop, X_train, y_train, X_test, y_test): #Evaluate each performance.
    fitness = []
    for w in pop:
        clf = MLPClassifier(learning_rate_init=1e-3, activation=w[0], solver = w[1], alpha=1e-5, hidden_layer_sizes=(int(w[2]), int(w[3])),  max_iter=500, n_iter_no_change=80)
        try:
            clf.fit(X_train, y_train)
            f = accuracy_score(clf.predict(X_test), y_test)

            fitness.append([f, clf, w])
        except:
            pass
    return fitness#


def ag_mlp(X_train, y_train, X_test, y_test, num_epochs = 10, size_mlp=10, prob_mut=0.8):
    pop = inicializacao_populacao_mlp(size_mlp) #Initial population.
    fitness = function_fitness_mlp(pop,  X_train, y_train, X_test, y_test) #Evaluate performacne of population.
    pop_fitness_sort = np.array(list(reversed(sorted(fitness,key=lambda x: x[0])))) #Resort performance ranking.
    
    for j in range(0, num_epochs): #num_epochs generations.
        length = len(pop_fitness_sort)
        #select pairs.
        parent_1 = pop_fitness_sort[:,2][:length//2]
        parent_2 = pop_fitness_sort[:,2][length//2:]

        #Crossover
        child_1 = [cruzamento_mlp(parent_1[i], parent_2[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = [cruzamento_mlp(parent_2[i], parent_1[i]) for i in range(0, np.min([len(parent_2), len(parent_1)]))]
        child_2 = mutacao_mlp(child_2, prob_mut)
        
        #Calculate each child's performance and add to population list.
        fitness_child_1 = function_fitness_mlp(child_1,X_train, y_train, X_test, y_test)
        fitness_child_2 = function_fitness_mlp(child_2, X_train, y_train, X_test, y_test)
        pop_fitness_sort = np.concatenate((pop_fitness_sort, fitness_child_1, fitness_child_2))
        sort = np.array(list(reversed(sorted(pop_fitness_sort,key=lambda x: x[0]))))
        
        #Resort performance of population and find the best individual.
        pop_fitness_sort = sort[0:size_mlp, :]
        best_individual = sort[0][1]
    
    return best_individual

SingleDataset = np.loadtxt('tictac_single.txt')
SingleX = SingleDataset[:,:9]
Singley = SingleDataset[:,9:]
X_trainSingle, X_testSingle, y_trainSingle, y_testSingle = train_test_split(SingleX, Singley, test_size=0.2, random_state=42)

clf = ag_mlp(X_trainSingle, y_trainSingle, X_testSingle, y_testSingle, num_epochs = 10, size_mlp=10, prob_mut=0.8)

print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
print("Results of GA-MLP model with single label.")
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

#Save trained model's parameters.
if not os.path.exists('TrainedParameters'):
    os.makedirs('TrainedParameters')
dump(clf, './TrainedParameters/GA-ClfMLP_singlelabel.joblib')


