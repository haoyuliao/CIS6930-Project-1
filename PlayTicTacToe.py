import numpy as np
import random
from sklearn.metrics import accuracy_score
from joblib import dump, load
import warnings

warnings.filterwarnings("ignore")

print('Please run the program in 64 bits version of python.')
print('Otherwise, the errors will be occured like ValueError: Buffer dtype mismatch...')

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

def showCurrentBoard(showBoard): #Show current borad condition.
    print("%s ｜ %s ｜ %s"%(showBoard[0],showBoard[1],showBoard[2]))
    print("－－－－－－")
    print("%s ｜ %s ｜ %s"%(showBoard[3],showBoard[4],showBoard[5]))
    print("－－－－－－")
    print("%s ｜ %s ｜ %s"%(showBoard[6],showBoard[7],showBoard[8]))

def winningCheck(showBoard): #Check who is winner.
    if showBoard[0]==showBoard[1] and showBoard[1]==showBoard[2]:
        return showBoard[0]
    if showBoard[3]==showBoard[4] and showBoard[4]==showBoard[5]:
        return showBoard[3]
    if showBoard[6]==showBoard[7] and showBoard[7]==showBoard[8]:
        return showBoard[6]
    if showBoard[0]==showBoard[3] and showBoard[3]==showBoard[6]:
        return showBoard[0]
    if showBoard[1]==showBoard[4] and showBoard[4]==showBoard[7]:
        return showBoard[1]
    if showBoard[2]==showBoard[5] and showBoard[5]==showBoard[8]:
        return showBoard[2]
    if showBoard[0]==showBoard[4] and showBoard[4]==showBoard[8]:
        return showBoard[0]
    if showBoard[2]==showBoard[4] and showBoard[4]==showBoard[6]:
        return showBoard[2]
    return None

#Load the all trained model.
ModelsName = ['ClfLinearSVM_multilabel',
              'ClfRBFSVM_multilabel',
              'ClfKNN_multilabel',
              'ClfMLP_multilabel',
              'ClfLinearSVM_singlelabel',
              'ClfRBFSVM_singlelabel',
              'ClfKNN_singlelabel',
              'ClfMLP_singlelabel',
              'RegKNN_multilabel',
              'RegMLP_multilabel',
              'RegLN_multilabel',
              'GA-ClfMLP_singlelabel']
model = [] 
modelList =[]
idx = 0
for mn in ModelsName: #Load all trained models' parameters.
    if mn == 'RegLN_multilabel':
        model.append(np.load('./GameParameters/'+mn+'.npy'))
    else:
        model.append(load('./GameParameters/'+mn+'.joblib'))
    modelList.append(idx)
    idx+=1

#Load liner gression of trained parameters.
LNregThresholds = np.load('./GameParameters/RegLN_multilabel_Thresholds.npy')
while True:
    try:
        playagain=input('\nWould you like to set up a new game? key Y/N: ') #Set up the game.
        if playagain != "Y" and playagain != "y" and playagain != "N" and playagain != "n":
            print('Please enter Y/N or enter crrect name of model.') 
            continue
        if playagain == "N" or playagain == "n":
                print('Bye-bye.')
                break
        idx = 0
        print('Current available model:') #Showing all current all available.
        for cm in ModelsName:
            if cm == 'ClfMLP_singlelabel': #The bestest model is ClfMLP_singlelabel.
                print('%s with key %s (bestest model)' %(cm, idx))
            else:
                print('%s with key %s' %(cm, idx))
            idx += 1
            
        InputPlyModel=int(input('\nWhich model you want to play with? key 0~11: ')) #Select model to play with.
        if playagain == "Y" or playagain == "y" and InputPlyModel in modelList:
            PlyModel = model[InputPlyModel] #Get weights from asigned model.
            currentBoard=np.array([0,0,0,0,0,0,0,0,0]) #Initial current borad condition.
            showBoard=[1,2,3,4,5,6,7,8,9] #Initial showing borad condition.
            showCurrentBoard(showBoard) #Show current borad condition.
            winner = None #Initial winner statement.
            while True:
                while True:
                    playerSteps=int(input('\nNow your turn(1~9): '))-1 #Player X move.
                    if showBoard[playerSteps] != 'X' and showBoard[playerSteps] != 'O':
                        showBoard[playerSteps]='X'
                        currentBoard[playerSteps] = 1
                        showCurrentBoard(showBoard)
                        break
                    else:
                        print('You cannot move the this place!!')
                        
                winner = winningCheck(showBoard) #Check whether has winner now.
                if  winner != None:
                    print('The winner is you.')
                    break
                if 0 not in currentBoard:
                    print("No winner no loser!!")
                    break
                
                print('Now my turn.')
                while True:
                    #print(ModelsName[InputPlyModel])
                    if ModelsName[InputPlyModel] == 'RegLN_multilabel':#For linear regression model.
                        optimalSteps = currentBoard @ PlyModel.T #Predict optimal steps for play O by LN model.
                        optimalSteps[optimalSteps >= LNregThresholds] = 1
                        optimalSteps[optimalSteps < LNregThresholds] = 0
                    else:
                        optimalSteps = PlyModel.predict(currentBoard.reshape(1, -1)) #Predict optimal steps by trained model.
                        
                    if np.max(optimalSteps.shape) > 1: #Check the model is trained by multi label or single label.
                        optimalSteps = optimalSteps.reshape(np.max(optimalSteps.shape),)
                        optimalSteps = np.array(value2binary(optimalSteps)) #If the model is regression, turn numeric results to binary results.
                        if 1 not in optimalSteps: #Check whether the optimal steps is located in current board.
                            optimalSteps = random.choice(list(np.where(currentBoard == 0)[0]))
                        else:
                            optimalSteps = random.choice(list(np.where(optimalSteps == 1)[0]))
                            if currentBoard[optimalSteps] != 0:
                                optimalSteps = random.choice(list(np.where(currentBoard == 0)[0]))
                    else:
                        optimalSteps = int(optimalSteps)
                        
                    if showBoard[optimalSteps] != 'X' and showBoard[optimalSteps] != 'O': #Player O move by optimal steps.
                        showBoard[optimalSteps]='O'
                        currentBoard[optimalSteps] = -1
                        showCurrentBoard(showBoard)
                        break
                    if 0 not in currentBoard:
                        print("No winner no loser!!")
                        break
                    
                winner = winningCheck(showBoard) #Check whether has winner now.
                if  winner != None:
                    print('The winner is me.')
                    break
                if 0 not in currentBoard:
                    break
        else:
            print('Please enter Y/N or enter crrect name of model.')
    except:
        continue

    
