import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Generate synthetic data
X, y_true = make_blobs(
    n_samples=600,
    n_features=5,
    centers=4,
    cluster_std=1.2,
    random_state=42
)

# Elbow & Silhouette
sse = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    sse.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

# Final model
k_opt = 4
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
final_labels = kmeans.fit_predict(X)

# PCA Visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels)
plt.title("K-Means Clustering with PCA")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()
