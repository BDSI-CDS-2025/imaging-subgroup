'''
clustering_by_dataset.py
For all of the different input feature sets, runs a K-Means clustering and saves the
results and visualizations to help with model assessment and interpretability.
'''

from pathlib import Path
from config import SETUP
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import seaborn as sns

input_datasets = [str(v['X']) for v in SETUP.values()]
clinical_data_path = r"C:\Users\k8pod\BDSICDS\clinicalData_clean.csv"
output_root = Path("results/tables/clustering_by_dataset")

# --- Load clinical data once ---
clinical_data = pd.read_csv(clinical_data_path)
predictors = ['ER', 'PR', 'HER2']
clinical_data = clinical_data.dropna(subset=predictors)

def entropy(counts):
    probs = counts / np.sum(counts)
    return -np.sum(probs * np.log2(probs + 1e-10))

id_cols = ["Patient ID", "Patient_ID", "Patient.ID", "patient_id", "ID"]

for feature_path in input_datasets:
    feature_name = Path(feature_path).stem
    print(f"\nProcessing feature set: {feature_name}")

    # Load and clean features
    features_df = pd.read_csv(feature_path).dropna()
    if "Unnamed: 0" in features_df.columns:
        imfeatures = features_df.drop(columns=["Unnamed: 0"])
    else:
        imfeatures = features_df

    # Find the Patient ID column
    merge_col = None
    for col in id_cols:
        if col in imfeatures.columns:
            merge_col = col
            break
    if merge_col is None:
        raise ValueError(f"No Patient ID column found for merging in {feature_path}. Columns: {imfeatures.columns.tolist()}")

    # Merge only ER, PR, HER2 from clinical data
    merged = imfeatures.merge(
        clinical_data[["Patient ID", "ER", "PR", "HER2"]],
        left_on=merge_col,
        right_on="Patient ID",
        how="inner"
    )

    # Drop all ID columns for clustering
    feature_cols = [col for col in imfeatures.columns if col not in id_cols]
    X = merged[feature_cols]

    # Only standardize if this is the imagingFeatures dataset
    if feature_name == "imagingFeatures":
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)
        standardized_df = pd.DataFrame(X_std, columns=feature_cols)
        out_dir = output_root / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)
        standardized_path = out_dir / "standardized_features.csv"
        standardized_df.to_csv(standardized_path, index=False)
    else:
        X_std = X.values  # Use as-is, already PCA-reduced
        out_dir = output_root / feature_name
        out_dir.mkdir(parents=True, exist_ok=True)

    # --- KMeans: Silhouette Score to select best k ---
    silhouette_scores = []
    cluster_results = {}
    for k in range(2, 15):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_std)
        silhouette = silhouette_score(X_std, labels)
        silhouette_scores.append(silhouette)
        cluster_results[k] = labels

    # Plot silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 15), silhouette_scores, marker='o')
    plt.title(f"Silhouette Scores for k-means ({feature_name})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(out_dir / "kmeans_silhouette_scores.png")
    plt.close()

    # Select best k for KMeans
    best_k_kmeans = np.argmax(silhouette_scores) + 2
    kmeans = KMeans(n_clusters=best_k_kmeans, random_state=42, n_init=10)
    merged['KMeans_Cluster'] = kmeans.fit_predict(X_std)
    joblib.dump(kmeans, out_dir / "kmeans_model.joblib")

    # --- GMM: BIC/AIC to select best k ---
    bics = []
    aics = []
    gmm_models = {}
    for k in range(2, 15):
        try:
            gmm = GaussianMixture(n_components=k, random_state=42, reg_covar=1e-3)
            gmm.fit(X_std)
            bics.append(gmm.bic(X_std))
            aics.append(gmm.aic(X_std))
            gmm_models[k] = gmm
        except ValueError as e:
            print(f"GMM failed for k={k}: {e}")
            # Fill with NaN so plotting works
            bics.append(np.nan)
            aics.append(np.nan)

    # Plot BIC/AIC
    plt.figure()
    plt.plot(range(2, 15), bics, label='BIC', marker='o')
    plt.plot(range(2, 15), aics, label='AIC', marker='o')
    plt.title(f"GMM Model Selection ({feature_name})")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_dir / "gmm_bic_aic.png")
    plt.close()

    # Select best k for GMM (lowest BIC)
    best_k_gmm = np.argmin(bics) + 2
    gmm = gmm_models[best_k_gmm]
    merged['GMM_Cluster'] = gmm.predict(X_std)
    joblib.dump(gmm, out_dir / "gmm_model.joblib")

    # --- Evaluation and saving ---
    merged['Mol_Subtype'] = merged['ER'].astype(str) + '/' + merged['PR'].astype(str) + '/' + merged['HER2'].astype(str)
    
    from scipy.stats import chi2_contingency
    # ...existing code...

    for predictor in predictors + ['Mol_Subtype']:
        pred_dir = out_dir / predictor
        pred_dir.mkdir(exist_ok=True)

        # KMeans crosstab
        table_kmeans = pd.crosstab(merged['KMeans_Cluster'], merged[predictor])
        chi2_kmeans, p_kmeans, _, _ = chi2_contingency(table_kmeans)
        table_kmeans.to_csv(pred_dir / "kmeans_crosstab.csv")
        (table_kmeans.apply(lambda r: r/r.sum(), axis=1) * 100).round(2).to_csv(pred_dir / "kmeans_crosstab_percent.csv")
        table_kmeans.apply(entropy, axis=1).to_csv(pred_dir / "kmeans_entropy.csv")
        # Save p-value
        with open(pred_dir / "kmeans_chi2_pvalue.txt", "w") as f:
            f.write(f"Chi-squared p-value: {p_kmeans}\n")
            if p_kmeans < 0.05:
                f.write("Significant at p < 0.05\n")
            else:
                f.write("Not significant at p < 0.05\n")

        # --- KMeans Heatmap ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(table_kmeans, annot=True, fmt='d', cmap='Blues')
        plt.title(f"KMeans Cluster vs {predictor} ({feature_name})\nChi2 p={p_kmeans:.3g}{' *' if p_kmeans < 0.05 else ''}")
        plt.ylabel("KMeans Cluster")
        plt.xlabel(predictor)
        plt.tight_layout()
        plt.savefig(pred_dir / "kmeans_heatmap.png")
        plt.close()

        # GMM crosstab
        table_gmm = pd.crosstab(merged['GMM_Cluster'], merged[predictor])
        chi2_gmm, p_gmm, _, _ = chi2_contingency(table_gmm)
        table_gmm.to_csv(pred_dir / "gmm_crosstab.csv")
        (table_gmm.apply(lambda r: r/r.sum(), axis=1) * 100).round(2).to_csv(pred_dir / "gmm_crosstab_percent.csv")
        table_gmm.apply(entropy, axis=1).to_csv(pred_dir / "gmm_entropy.csv")
        # Save p-value
        with open(pred_dir / "gmm_chi2_pvalue.txt", "w") as f:
            f.write(f"Chi-squared p-value: {p_gmm}\n")
            if p_gmm < 0.05:
                f.write("Significant at p < 0.05\n")
            else:
                f.write("Not significant at p < 0.05\n")

        # --- GMM Heatmap ---
        plt.figure(figsize=(8, 6))
        sns.heatmap(table_gmm, annot=True, fmt='d', cmap='Oranges')
        plt.title(f"GMM Cluster vs {predictor} ({feature_name})\nChi2 p={p_gmm:.3g}{' *' if p_gmm < 0.05 else ''}")
        plt.ylabel("GMM Cluster")
        plt.xlabel(predictor)
        plt.tight_layout()
        plt.savefig(pred_dir / "gmm_heatmap.png")
        plt.close()
    print(f"Results saved to {out_dir}")

print("All clustering and evaluations complete.")