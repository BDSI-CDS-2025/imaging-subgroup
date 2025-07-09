'''
total_corr.py
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
path = Path().cwd()
IMG_PATH = path / "data" / "raw" / "imagingFeatures.csv"
OUTPUT_PATH = path / "results" / "figures"
img = pd.read_csv(IMG_PATH).drop(['Unnamed: 0', 'Patient ID'], axis=1).dropna()

# Drop non-feature columns if present (e.g., Patient ID)
img = img.loc[:, img.columns != 'Patient ID']

# Compute correlation matrix
corr_matrix = img.corr()

# Plot heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, square=True, cbar_kws={"shrink": 0.5})
plt.title('Correlation Matrix Heatmap (All Features)')
plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'correlation_heatmap_all_features.png', dpi=300)
plt.close()

# Plot clustered heatmap
sns.set(font_scale=0.6)
g = sns.clustermap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    figsize=(20, 20),
    cbar_kws={"shrink": 0.5},
    method='average',  # linkage method
    metric='euclidean',  # distance metric
)
plt.suptitle('Hierarchical Clustering of Feature Correlations', y=1.02)
plt.savefig(OUTPUT_PATH / 'clustermap_correlation_all_features.png', dpi=300, bbox_inches='tight')
plt.close()

# (Optional) Save correlation matrix as CSV
corr_matrix.to_csv(OUTPUT_PATH / 'correlation_matrix_all_features.csv')