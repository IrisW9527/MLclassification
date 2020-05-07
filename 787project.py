import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import timeit
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


data = np.genfromtxt('winequality.csv', delimiter=',', skip_header=1)

X = data[:,:-1]
Y = data[:,-1]

mean_X = np.mean(X, axis = 0)
std_X = np.std(X, axis = 0)
X_centered = (X - mean_X)/std_X

X_train, X_test, Y_train, Y_test = train_test_split(X_centered, Y, test_size=0.25)

print('# of 1 in Y_test: ', np.count_nonzero(Y_test == 1))
print('# of 0 in Y_test: ', (Y_test == 0).sum())


# KNN
knn = KNeighborsClassifier(n_neighbors=13)
# accuracy = []
startTime1 = timeit.default_timer()
knn.fit(X_train, Y_train)
stopTime1 = timeit.default_timer()
print('\ntraining time = ', stopTime1 - startTime1)

startTime2 = timeit.default_timer()
prediction = knn.predict(X_test)
stopTime2 = timeit.default_timer()
print('testing time = ', stopTime2 - startTime2)

accuracy = metrics.accuracy_score(Y_test, prediction)


# # AdaBoost
# ab = AdaBoostClassifier()
#
# startTime1 = timeit.default_timer()
# ab.fit(X_train, Y_train)
# stopTime1 = timeit.default_timer()
# print('\ntraining time = ', stopTime1 - startTime1)
#
# startTime2 = timeit.default_timer()
# prediction = ab.predict(X_test)
# stopTime2 = timeit.default_timer()
# print('testing time = ', stopTime2 - startTime2)



# # SVM
# svm = SVC(kernel='rbf', gamma=0.01, C=0.1, probability=True) # " Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’ "
#
# startTime1 = timeit.default_timer()
# svm.fit(X_train, Y_train)
# stopTime1 = timeit.default_timer()
# print('\ntraining time = ', stopTime1 - startTime1)
#
# startTime2 = timeit.default_timer()
# prediction = svm.predict(X_test)
# stopTime2 = timeit.default_timer()
# print('testing time = ', stopTime2 - startTime2)


# # accuracy & confusion matrix
# accuracy = metrics.accuracy_score(Y_test, prediction)
# print('accuracy: ', accuracy)
# #
# # print('\nconfusion matrix: ')
# # tn, fp, fn, tp = confusion_matrix(Y_test, prediction).ravel()
# # print('tn, fp, fn, fp: ', tn, fp, fn, tp)


# # ROC curve
# probs = svm.predict_proba(X_test)
# # probs = ab.predict_proba(X_test)
# # probs = knn.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
# roc_auc = metrics.auc(fpr, tpr)
# print('roc_auc = ', roc_auc)
#
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()



print('done')