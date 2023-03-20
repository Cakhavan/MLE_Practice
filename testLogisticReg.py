import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from logisticRegression import LogisticRegression

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/len(y_true)

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
train_acc = accuracy(y_train, y_pred)
print('training acc: ', train_acc)

y_pred = clf.predict(X_test)
test_acc = accuracy(y_test, y_pred)
print('test acc: ', test_acc)