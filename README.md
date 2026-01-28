# K-Means Clustering on Synthetic Data

## 📌 Project Overview
This project demonstrates the implementation and evaluation of the K-Means clustering algorithm on a synthetic dataset. The objective is to understand how centroid-based clustering works and how to determine the optimal number of clusters using evaluation metrics.

---

## 📊 Dataset Description
- Dataset generated using `sklearn.datasets.make_blobs`
- Number of samples: 600
- Number of features: 5
- Number of clusters: 4 (ground truth)

The dataset is synthetic and designed to clearly represent distinct clusters.

---

## 🧠 Methodology
1. Generated synthetic multi-dimensional data using `make_blobs`
2. Applied K-Means clustering for multiple values of K (from 2 to 10)
3. Used the **Elbow Method (SSE)** to identify the optimal number of clusters
4. Used **Silhouette Score** to validate cluster quality
5. Selected **K = 4** as the optimal number of clusters
6. Applied **Principal Component Analysis (PCA)** to reduce the data to 2D
7. Visualized the final clusters

---

## 📈 Evaluation Metrics
- **Sum of Squared Errors (SSE)** – Elbow Method
- **Silhouette Score** – Cluster separation quality

Both metrics confirmed that K = 4 provides the best clustering structure.

---

## 📊 Results
- Clear elbow observed at K = 4
- Highest silhouette score around K = 4
- PCA visualization shows well-separated clusters
- Final centroids closely match the true cluster centers

---

## ▶ How to Run the Project

```bash
pip install -r requirements.txt
python src/kmeans_clustering.py
