# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 15:18:05 2024

@author: Shibo
"""

#Step 1: Retrieve and Load the Olivetti Faces Dataset
from sklearn.datasets import fetch_olivetti_faces

# Load the dataset
data = fetch_olivetti_faces()
images = data.images
targets = data.target

print(f"Number of images: {images.shape[0]}")
print(f"Image shape: {images.shape[1:]}")


#Step 2: Split the Dataset

from sklearn.model_selection import train_test_split

# Split the data into training and temp sets
X_train, X_temp, y_train, y_temp = train_test_split(images, targets, test_size=0.3, stratify=targets, random_state=42)

# Split the temp set into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

#Step 3: Train a Classifier using K-Fold Cross Validation
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Flatten the images for training
X_train_flat = X_train.reshape(X_train.shape[0], -1)

# Initialize the classifier
clf = SVC(kernel='linear', random_state=42)

# Perform k-fold cross-validation
scores = cross_val_score(clf, X_train_flat, y_train, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean cross-validation score: {scores.mean()}")

#Step 4: Dimensionality Reduction using Hierarchical Clustering
# use Agglomerative Hierarchical Clustering (AHC) with different similarity measures
#a)  Euclidean Distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn import metrics

# Apply Agglomerative Clustering with Euclidean distance
ahc_euclidean = AgglomerativeClustering(n_clusters=40, metric='euclidean', linkage='ward')
ahc_euclidean.fit(X_train_flat)

# Calculate silhouette score
labels_euclidean = ahc_euclidean.labels_
silhouette_avg_euclidean = silhouette_score(X_train_flat, labels_euclidean)
print(f"Silhouette score (Euclidean): {silhouette_avg_euclidean}")

#b) Minkowski Distance
# Apply Agglomerative Clustering with Minkowski distance
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_flat)
minkowski_dist_matrix = pairwise_distances(X_scaled, metric='minkowski', p=3)  # p=3 for Minkowski distance

ahc_minkowski = AgglomerativeClustering(n_clusters=40, metric='precomputed', linkage='average')
ahc_minkowski.fit(minkowski_dist_matrix)

# Calculate silhouette score
labels_minkowski = ahc_minkowski.labels_
silhouette_avg_minkowski = silhouette_score(minkowski_dist_matrix, labels_minkowski, metric='precomputed')
print(f"Silhouette score (Minkowski): {silhouette_avg_minkowski}")

#c) Cosine Similarity
# Apply Agglomerative Clustering with Cosine similarity
ahc_cosine = AgglomerativeClustering(n_clusters=40, metric='cosine', linkage='average')
ahc_cosine.fit(X_train_flat)

# Calculate silhouette score
labels_cosine = ahc_cosine.labels_
silhouette_avg_cosine = silhouette_score(X_train_flat, labels_cosine)
print(f"Silhouette score (Cosine): {silhouette_avg_cosine}")


#Step 5: Choose the Number of Clusters
# Determine the optimal number of clusters based on silhouette scores
def calculate_silhouette_scores(X, metric, linkage, distance_matrix=None):
    silhouette_scores = []
    for n_clusters in range(2, 150):  
        if distance_matrix is not None:
            ahc = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
            labels = ahc.fit_predict(distance_matrix)
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        else:
            ahc = AgglomerativeClustering(n_clusters=n_clusters, metric=metric, linkage=linkage)
            labels = ahc.fit_predict(X)
            score = silhouette_score(X, labels, metric=metric)
        silhouette_scores.append(score)
        print(f"Number of clusters: {n_clusters}, Silhouette score: {score}")
    return silhouette_scores

# Calculate silhouette scores for Euclidean distance
print("Euclidean Distance:")
silhouette_scores_euclidean = calculate_silhouette_scores(X_scaled, 'euclidean', 'ward')
import numpy as np
optimal_clusters_euclidean = np.argmax(silhouette_scores_euclidean)+2
print(f"Optimal number of clusters (Euclidean): {optimal_clusters_euclidean}")

# Calculate silhouette scores for Minkowski distance
print("\nMinkowski Distance:")
silhouette_scores_minkowski = calculate_silhouette_scores(X_scaled, 'precomputed', 'average', minkowski_dist_matrix)
optimal_clusters_minkowski = np.argmax(silhouette_scores_minkowski) +2
print(f"Optimal number of clusters (Minkowski): {optimal_clusters_minkowski}")

# Calculate silhouette scores for Cosine similarity
print("\nCosine Similarity:")
silhouette_scores_cosine = calculate_silhouette_scores(X_scaled, 'cosine', 'average')
optimal_clusters_cosine = np.argmax(silhouette_scores_cosine) + 2
print(f"Optimal number of clusters (Cosine): {optimal_clusters_cosine}")


#6 Use the set from (4(a), 4(b), or 4(c)) to train a classifier as in (3) using k-fold cross validation.
# Apply Agglomerative Clustering with the optimal number of clusters
ahc_optimal1 = AgglomerativeClustering(n_clusters=optimal_clusters_euclidean, metric='euclidean', linkage='ward')
ahc_optimal1.fit(X_train_flat)

# Get the cluster labels
labels_euclidean = ahc_optimal1.labels_

# You can use these labels to reduce the dimensionality or for further analysis
# For example, you can use the labels to create a new feature set
X_reduced = np.zeros((X_train_flat.shape[0], optimal_clusters_euclidean))
for i in range(optimal_clusters_euclidean):
    X_reduced[:, i] = (labels_euclidean == i).astype(int)

print(f"Reduced feature set shape: {X_reduced.shape}")

# Train the classifier on the reduced dataset
scores_reduced = cross_val_score(clf, X_reduced, y_train, cv=5)

print(f"Cross-validation scores on reduced dataset1: {scores_reduced}")
print(f"Mean cross-validation score on reduced dataset1: {scores_reduced.mean()}")

#2
ahc_optimal2 = AgglomerativeClustering(n_clusters=optimal_clusters_minkowski, metric='precomputed', linkage='average')
ahc_optimal2.fit(minkowski_dist_matrix)

# Get the cluster labels
labels_minkowski = ahc_optimal2.labels_

# You can use these labels to reduce the dimensionality or for further analysis
# For example, you can use the labels to create a new feature set
X_reduced2 = np.zeros((minkowski_dist_matrix.shape[0], optimal_clusters_minkowski))
for i in range(optimal_clusters_minkowski):
    X_reduced2[:, i] = (labels_minkowski == i).astype(int)

print(f"Reduced feature set shape: {X_reduced2.shape}")

# Train the classifier on the reduced dataset
scores_reduced2 = cross_val_score(clf, X_reduced2, y_train, cv=5)

print(f"Cross-validation scores on reduced dataset2: {scores_reduced2}")
print(f"Mean cross-validation score on reduced dataset2: {scores_reduced2.mean()}")

#3
ahc_optimal3 = AgglomerativeClustering(n_clusters=optimal_clusters_cosine, metric='cosine', linkage='average')
ahc_optimal3.fit(X_train_flat)

# Get the cluster labels
labels_cosine = ahc_optimal3.labels_

# You can use these labels to reduce the dimensionality or for further analysis
# For example, you can use the labels to create a new feature set
X_reduced3 = np.zeros((X_train_flat.shape[0], optimal_clusters_cosine))
for i in range(optimal_clusters_cosine):
    X_reduced3[:, i] = (labels_cosine == i).astype(int)

print(f"Reduced feature set shape: {X_reduced3.shape}")

# Train the classifier on the reduced dataset
scores_reduced3 = cross_val_score(clf, X_reduced3, y_train, cv=5)

print(f"Cross-validation scores on reduced dataset3: {scores_reduced3}")
print(f"Mean cross-validation score on reduced dataset3: {scores_reduced3.mean()}")