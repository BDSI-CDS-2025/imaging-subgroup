# Clustering for each of 10 variable groups with/without clinical data 
# Using one-hot encoding and my config_FINAL to predict ER, PR, HER2 !
# HOORAY! :)
'''
import os
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
'''


from pathlib import Path
from config_FINAL import SETUP
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from functools import reduce
from sklearn.model_selection import StratifiedKFold

from config_per_var_fam import SET  # <-- Use your variable family config here

input_datasets = [str(v['X']) for v in SET.values()]

clinical_data_path = r"C:\Users\k8pod\BDSICDS\clinicalData_clean.csv"
output_root = Path("results/tables/clustering_by_dataset")

# --- Load clinical data once ---
clinical_data = pd.read_csv(clinical_data_path)
predictors = ['ER', 'PR', 'HER2']
clinical_data = clinical_data.dropna(subset=predictors)

# --- Patient ID columns to check ---
id_cols = ['Patient ID', 'patient_id', 'ID', 'Patient.ID']

cluster_labels_all = []

for feature_path in input_datasets:
    feature_name = Path(feature_path).stem
    features_df = pd.read_csv(feature_path).dropna()
    # Find Patient ID column
    merge_col = next((col for col in id_cols if col in features_df.columns), None)
    if not merge_col:
        raise ValueError(f"No Patient ID column found in {feature_path}")
    merged = features_df.merge(clinical_data[['Patient ID'] + predictors], left_on=merge_col, right_on='Patient ID', how='inner')
    feature_cols = [col for col in features_df.columns if col not in id_cols]
    X = merged[feature_cols].values

     # --- Standardize ALL datasets ---
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    out_dir = output_root / feature_name
    out_dir.mkdir(parents=True, exist_ok=True)
    standardized_df = pd.DataFrame(X_std, columns=feature_cols)
    standardized_path = out_dir / "standardized_features.csv"
    standardized_df.to_csv(standardized_path, index=False)

    # --- KMeans optimal k ---
    silhouette_scores = []
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_std)
        silhouette_scores.append(silhouette_score(X_std, labels))
    best_k_kmeans = np.argmax(silhouette_scores) + 2
    kmeans = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_std)

    # --- GMM optimal k ---
    bics = []
    gmm_models = {}
    for k in range(2, 15):
        gmm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-3)
        gmm.fit(X_std)
        bics.append(gmm.bic(X_std))
        gmm_models[k] = gmm
    best_k_gmm = np.argmin(bics) + 2
    gmm = gmm_models[best_k_gmm]
    gmm_labels = gmm.predict(X_std)

    # Store cluster labels for each sample (can use either or both methods)
    cluster_labels_all.append(pd.DataFrame({
        'Patient ID': merged['Patient ID'],
        f'{feature_name}_kmeans': kmeans_labels,
        f'{feature_name}_gmm': gmm_labels
    }))

# --- Merge all cluster labels on Patient ID ---
cluster_features = reduce(lambda left, right: pd.merge(left, right, on='Patient ID', how='inner'), cluster_labels_all)
full_data = pd.merge(cluster_features, clinical_data[['Patient ID'] + predictors], on='Patient ID', how='inner')

# --- One-hot encode cluster labels ---
cluster_cols = [col for col in full_data.columns if col not in ['Patient ID'] + predictors]
encoder = OneHotEncoder(sparse_output=False)
X_clusters = encoder.fit_transform(full_data[cluster_cols])
y_dict = {marker: full_data[marker] for marker in predictors}

# --- Train classifier and compute AUC for each marker ---
from config_FINAL import SETUP

inputs = [str(v['X']) for v in SETUP.values()]

results = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for feature_path in inputs:
    feature_name = Path(feature_path).stem
    features_df = pd.read_csv(feature_path).dropna()
    # Find Patient ID column
    merge_col = next((col for col in id_cols if col in features_df.columns), None)
    if not merge_col:
        raise ValueError(f"No Patient ID column found in {feature_path}")
    merged = features_df.merge(clinical_data[['Patient ID'] + predictors], left_on=merge_col, right_on='Patient ID', how='inner')
    feature_cols = [col for col in features_df.columns if col not in id_cols]
    X = merged[feature_cols].values
    y_dict = {marker: merged[marker] for marker in predictors}

    # Standardize
    if inputs[feature_path] != "PC1_3.csv" or inputs[feature_path] != "PC1_3_with_clin.csv" or inputs[feature_path] != "PC1.csv" or inputs[feature_path] != "PC1_3_with_clin.csv" or inputs[feature_path] != "PC1_with_clin.csv" or inputs[feature_path] != "PC_90.csv" or inputs[feature_path] != "PC_90_with_clin.csv" or inputs[feature_path] != "VARS_IN_PC1_with_clin.csv" or inputs[feature_path] != "VARS_IN_PC1.csv" or inputs[feature_path] != "VARS_IN_PC_1_3_with_clin.csv" or inputs[feature_path] != "VARS_IN_PC_1_3.csv" or inputs[feature_path] != "VARS_IN_PC_90_with_clin.csv" or inputs[feature_path] != "VARS_IN_PC_90.csv":
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

    # Cross-validated AUCs for each marker
    for marker in predictors:
        y = y_dict[marker]
        aucs = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X_std, y):
            X_train, X_test = X_std[train_idx], X_std[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_train, y_train)
            y_pred = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)
        results[f"{feature_name}_{marker}"] = {
            "mean_auc": np.mean(aucs),
            "std_auc": np.std(aucs),
            "fold_aucs": aucs
        }

print("Cross-validated AUCs for config_FINAL datasets:", results)
    
    
    
'''   
    y = y_dict[marker]
    aucs = []
    for train_idx, test_idx in skf.split(X_clusters, y):
        X_train, X_test = X_clusters[train_idx], X_clusters[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)
        aucs.append(auc)
    results[marker] = {
        "mean_auc": np.mean(aucs),
        "std_auc": np.std(aucs),
        "fold_aucs": aucs
    }

print("Cross-validated AUCs:", results)
'''