import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from linearRegression import LinearRegression
import matplotlib.pyplot as plt

def accuracy(y_true, y_pred):
    return np.sum((y_true - y_pred)**2)/ len(y_true)

iris = datasets.load_iris()
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

clf = LinearRegression(iters=100, lr=0.01)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('test MSE: ', accuracy(y_test, y_pred))


y_pred_line = clf.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()