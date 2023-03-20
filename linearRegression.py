import numpy as np

def mse(y_pred, y_true):
    return (np.sum(y_pred - y_true)**2)/len(y_pred)

class LinearRegression:

    def __init__(self, lr=0.001, iters=10000):
        self.lr = lr
        self.iters = iters
        self.weights = None
        self.bias = None


    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        error = 1

        for _ in range(self.iters):
            y_pred = self.predict(X_train)
            dw = (1/n_samples)*np.dot(X_train.T, (y_pred-y_train))
            db = (1/n_samples) * np.sum(y_pred-y_train)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db

        
        print('training complete with MSE ', mse(self.predict(X_train), y_train))


    def predict(self, x):
        y_pred = np.dot(x, self.weights) + self.bias
        return y_pred
    
