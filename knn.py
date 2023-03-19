from collections import Counter
import numpy as np
import math

def euclidianDistance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))
    

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


    def predict(self, X_test):
        predictied_labels = [self._predict(x_test) for x_test in X_test]
        return predictied_labels

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidianDistance(x, x_train) for x_train in self.X_train]

        # Sort by distance and return indices of the first k neighbors
        k_idx = np.argsort(distances)[:self.k]

        # Grab labels of top k neighbors
        k_labels = [self.y_train[k] for k in k_idx]

        # Return most frequent neighbor class
        top_class = Counter(k_labels).most_common(1)[0][0]

        return top_class

    