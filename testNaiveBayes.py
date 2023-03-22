import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from naiveBayes import NaiveBayes

def accuracy(y_pred, y_true):
     return np.sum(y_pred == y_true) / len(y_true)

X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

clf = NaiveBayes()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
train_acc = accuracy(y_pred, y_train)
print('training accuracy: ', train_acc)

y_pred = clf.predict(X_test)
test_acc = accuracy(y_pred, y_test)
print('testing accuracy: ', test_acc)