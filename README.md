# K-Means Clustering Analysis from Scratch (Python)

A complete **from-scratch implementation of the K-Means clustering algorithm**, designed to analyze and compare **Random Initialization** and **K-Means++ Initialization** using experimental evaluation on the Iris dataset.

This project focuses on **algorithmic clarity, reproducibility, and experimental rigor**, rather than relying on high-level ML libraries.

---

## Overview

K-Means is highly sensitive to centroid initialization.  
This project empirically demonstrates:

- How different initialization strategies affect convergence
- Why K-Means++ provides more stable and lower-inertia solutions
- How to evaluate clustering performance through repeated experiments

All core logic is implemented manually using NumPy.

---

## Features

- K-Means clustering implemented from scratch
- Random and K-Means++ initialization strategies
- Multiple experimental runs to reduce randomness bias
- Inertia (WCSS) used as the evaluation metric
- Elbow method to estimate optimal number of clusters
- Modular, research-style project structure
- Reproducible results and saved plots

---

## Dataset

- **Dataset**: Iris
- **Samples**: 150
- **Features**: 4 numerical features
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- **Note**: Class labels are intentionally excluded (unsupervised learning)

The dataset is generated programmatically using `sklearn.datasets.load_iris` and saved as a clean numeric CSV file.

---

## Algorithms Implemented

### K-Means Clustering
- Euclidean distance metric
- Iterative centroid updates
- Convergence based on centroid stability

### Initialization Strategies
- **Random Initialization**
- **K-Means++ Initialization** (distance-weighted centroid selection)

### Evaluation
- Inertia (Within-Cluster Sum of Squares)
- Multiple independent runs per configuration
- Boxplot-based comparison
- Elbow plot for cluster selection

---

## Project Structure

```text
kmeans-analysis/
│
├── src/
│   ├── kmeans.py           # Core K-Means algorithm
│   ├── initialization.py  # Random & K-Means++ initialization
│   ├── metrics.py         # Inertia computation
│   ├── experiments.py     # Experiment runner & plotting
│   └── utils.py           # CSV loading utilities
│
├── data/
│   └── iris.csv            # Clean numeric dataset
│
├── results/
│   └── plots/              # Generated plots
│
├── make_iris.py            # Dataset generation script
├── README.md
└── .gitignore
