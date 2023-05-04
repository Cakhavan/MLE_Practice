import numpy as np
from sklearn import datasets as sk
from matplotlib import pyplot as plt

def euclidian_distance(p1, p2):
    return np.sqrt(np.sum((p1-p2)**2))

class KMeans():
    def __init__(self, K=3, n_iters=100):
        self.K = K
        self.n_iters = n_iters
    
    def predict(self, X):
        self.X = X
        n_samples, n_features = X.shape
        
        centroids = X[np.random.choice(n_samples, self.K, replace=False)]
        for _ in range(self.n_iters):
            clusters = [[] for _ in range(self.K)]
            self.create_clusters(clusters, centroids)

            centroids = self.update_centroids(clusters)
        return [self.get_labels(clusters, n_samples), centroids]

    def get_labels(self, clusters, n_samples):
        y = np.zeros(n_samples)
        for i in range(len(clusters)):
            for idx in clusters[i]:
                y[idx] = i
        return y
    
    def update_centroids(self, clusters):
        new_centroids = [[] for _ in range(self.K)]
        for i in range(self.K):
            new_centroids[i] = np.mean(self.X[clusters[i]], axis=0)
        return new_centroids
    
    def create_clusters(self, clusters, centroids):
        for idx, sample in enumerate(self.X):
            cluster_idx = self.closest_cluster(sample, centroids)
            clusters[cluster_idx].append(idx)
    
    def closest_cluster(self, sample, centroids):
        dist = [0]*self.K
        for i in range(len(centroids)):
            dist[i] = euclidian_distance(sample, centroids[i])
        return np.argmin(dist)


data = sk.make_blobs(n_samples=500, n_features=2, centers=5)
X, y = data
clf = KMeans(K=5, n_iters=500)
y, centroids = clf.predict(X)

plt.scatter(X[:,0], X[:,1], c=y)
plt.scatter([d[0] for d in centroids], [d[1] for d in centroids], c='black', marker='x')
plt.show()
    










