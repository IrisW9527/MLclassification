# MLclassification

In this project, the 1500 observations in “WineQuality” dataset is used where there are 11 input
variables and 1 output variable that is classified into 0s and 1s. Since the input variables are all
numerical data, the three classifiers adopted are:
- Support Vector machine
- K Nearest Neighbors
- AdaBoost
Performances are measured by model accuracies, confusion matrix, ROC curves, and AUC.

The dataset is firstly mean-centered and scaled to unit length. In order to validate the models
built, the dataset is partitioned into training data (1125 observations) and testing data (375
observations) by using “train_test_split” from scikit-learn. “train_test_split” helps split the data
into N folds and train on (N-1) and test on the rest data which is what cross-validation does.
