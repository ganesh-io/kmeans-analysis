Results and Observations

Experiments were conducted on the Iris dataset (150 samples, 4 features).

K-Means clustering was evaluated for k = 2, 3, and 4.

For each k, clustering was repeated multiple times using:

Random initialization

K-Means++ initialization

Observations:

K-Means++ consistently achieved lower and more stable inertia compared to random initialization.

Random initialization showed higher variance across runs due to sensitivity to initial centroid placement.

The elbow plot indicates a clear elbow around k = 3, aligning with the known structure of the Iris dataset.

This confirms that K-Means++ improves convergence stability and clustering quality.