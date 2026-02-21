import numpy as np

def inertia(X, centroids, clusters):
    total = 0.0
    for i, x in enumerate(X):
        total += np.linalg.norm(x - centroids[clusters[i]]) ** 2
    return total