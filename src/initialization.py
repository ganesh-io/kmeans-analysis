import numpy as np

def random_init(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def kmeans_plus_plus(X, k):
    centroids = []
    centroids.append(X[np.random.randint(len(X))])

    for _ in range(1, k):
        distances = np.array([
            min(np.linalg.norm(x - c) ** 2 for c in centroids)
            for x in X
        ])

        probabilities = distances / distances.sum()
        cumulative_probs = np.cumsum(probabilities)
        r = np.random.rand()

        for idx, prob in enumerate(cumulative_probs):
            if r < prob:
                centroids.append(X[idx])
                break

    return np.array(centroids)