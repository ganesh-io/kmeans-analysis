import numpy as np

class KMeans:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X, centroids):
        for _ in range(self.max_iters):
            clusters = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, clusters)

            shift = np.linalg.norm(new_centroids - centroids)
            if shift < self.tol:
                break

            centroids = new_centroids

        return centroids, clusters

    def _assign_clusters(self, X, centroids):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, clusters):
        centroids = []
        for i in range(self.k):
            cluster_points = X[clusters == i]
            if len(cluster_points) == 0:
                centroids.append(np.zeros(X.shape[1]))
            else:
                centroids.append(cluster_points.mean(axis=0))
        return np.array(centroids)