import numpy as np
import matplotlib.pyplot as plt

from kmeans import KMeans
from initialization import random_init, kmeans_plus_plus
from metrics import inertia
from utils import load_csv


def run_experiment(X, k, init_method, runs=5):
    inertias = []

    for _ in range(runs):
        if init_method == "random":
            centroids = random_init(X, k)
        else:
            centroids = kmeans_plus_plus(X, k)

        model = KMeans(k)
        final_centroids, clusters = model.fit(X, centroids)
        inertias.append(inertia(X, final_centroids, clusters))

    return inertias


def plot_results(random_vals, kpp_vals, k):
    plt.boxplot([random_vals, kpp_vals], tick_labels=["Random", "K-Means++"])
    plt.ylabel("Inertia")
    plt.title(f"K-Means Initialization Comparison (k={k})")
    plt.savefig(f"results/plots/k_{k}_comparison.png")
    plt.close()


def elbow_plot(X, max_k=8):
    inertias = []

    for k in range(1, max_k + 1):
        centroids = kmeans_plus_plus(X, k)
        model = KMeans(k)
        final_centroids, clusters = model.fit(X, centroids)
        inertias.append(inertia(X, final_centroids, clusters))

    plt.plot(range(1, max_k + 1), inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.savefig("results/plots/elbow_plot.png")
    plt.close()


if __name__ == "__main__":
    DATA_PATH = "data/iris.csv"
    X = load_csv(DATA_PATH)

    print("Dataset loaded:", X.shape)

    # 🔹 Run for k = 2, 3, 4
    for k in [2, 3, 4]:
        random_vals = run_experiment(X, k, "random")
        kpp_vals = run_experiment(X, k, "kpp")

        plot_results(random_vals, kpp_vals, k)
        print("Experiment completed for k =", k)

    # 🔹 Elbow plot
    elbow_plot(X)
    print("Elbow plot generated")