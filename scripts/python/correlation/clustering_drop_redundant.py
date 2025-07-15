'''
clustering_drop_redundant.py

Hierarchical clustering groups similar data points (rows or columns) based
on a chosen distance or similarity measure (Euclidean distance or Pearson correlation
coefficient)
'''

import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from pathlib import Path
import numpy as np

# Load data
# Load data
path = Path().cwd()
IMG_PATH = path / "data" / "raw" / "imagingFeatures.csv"
OUTPUT_PATH = path / "results" / "figures"
OUTPUT_DATA_PATH = path / "data" / "interim"
img = pd.read_csv(IMG_PATH)

# Compute correlation matrix
corr_matrix = img.drop(['Unnamed: 0', 'Patient ID'], axis=1).dropna().corr().abs()

# Convert correlation to distance
distance_matrix = 1 - corr_matrix
# Fix floating point errors: clip to [0, 1]
distance_matrix = distance_matrix.clip(lower=0, upper=1)

# Set diagonal to zero
np.fill_diagonal(distance_matrix.values, 0)

# Convert to condensed distance matrix for linkage
condensed_dist = squareform(distance_matrix.values, checks=False)

# Hierarchical clustering
# linkage matrix has four columns per merge:
# [idx1, idx2, distance, new_cluster_size]
Z = linkage(condensed_dist, method='average')

# Choose a threshold for cluster formation (e.g., 0.1 = features with corr > 0.9 are grouped)
threshold = 0.1
clusters = fcluster(Z, threshold, criterion='distance')

# For each cluster, keep only one feature (the first one)
features_to_keep = []
for cluster_id in set(clusters):
    members = [img.columns[i] for i, c in enumerate(clusters) if c == cluster_id]
    features_to_keep.append(members[0])  # keep the first feature in the cluster

# Create reduced dataframe
img_reduced = img[features_to_keep].drop(['Unnamed: 0'], axis = 1)
img_reduced.to_csv(OUTPUT_DATA_PATH / 'imagingFeatures_uncorrelated_clustering.csv', index=False)
print(f'img_reduced: {img_reduced.shape}')