import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from knn import KNN

def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred)/ len(y_true)

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])


iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
# print(len(X_train), len(X_test))

clf = KNN(3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('test accuracy: ', accuracy(y_test, y_pred))