# CIS6930-Project1-Tic-Tac-Toe
* This poject try to train AI with different algorithms to play Tic Tac Toe.

## Availabe Algorithms
* Classificaitons
 1. KNN
 2. Linear-SVM
 3. RBF-SVM
 4. MLP
 5. GA-MLP
* Regressions
 1. KNN
 2. Linear regression
 3. MLP
 
## Programs:
 1. Classiﬁers.py: classifier program for all classification models
 2. Regressors.py: regressor program for all regression models
 3. PlayTicTacToe.py: play Tic Tac Toe with all trained models.
 4. GA_ClfMLP_singleLabel.py: implement genetic algorithms to find best MLP parameters.
 
## Folders:
 1. TrainedCVresults: all trained CV records with different parameters.
 2. TrainedParameters: all trained models' parameters.
 3. RecordConfusionMatrix&Accuracy: all trained models' accuracy and confusion matrix.
 4. GameParameters: trained models' with the best parameters for playing Tic Tac Toe.
 
## Archives:
 1. ClassifiersRes4AllModels.xslx: all trained models' training accuracy and testing accuracy.
 2. RegressorsRes4AllModels.xslx:  all trained models' training accuracy and testing accuracy.
 3. Noise_ClassifiersRes4AllModels.xslx:  all trained models' training accuracy and testing accuracy with adding noise.
 4. Reduce9by10_ClassifiersRes4AllModels.xslx:  all trained models' training accuracy and testing accuracy with reducing dataset.
 5. GA_ClfMLP_singleLabelResults.xlsx: Results of GA-MLP classification model for single model.
 
## Notes
* Please run the PlayTicTacToe.py program in 64 bits version of python. (Important.)
* Otherwise, the errors will be occured like (ValueError: Buffer dtype mismatch...)
