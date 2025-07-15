"""
final_produce_custom_figures.py

This script reads in a CSV file (formatted like summary_auc.csv) containing model
statistics, reshapes it so that each model is in its own row, and produces figures
based on user-selected datasets, models, and predictors. It also writes out a filtered
CSV for reference.

User-adjustable lists:
    SELECTED_DATASETS: List of datasets to include.
    SELECTED_MODELS: List of models to include.
    SELECTED_PREDICTORS: List of predictors (outcomes) to include.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------
# User adjustable parameters - modify these as needed
SELECTED_DATASETS = ["ALL_IMG",
                     "ALL_IMG_with_clin",
                     "PC1_3",
                     "VARS_IN_PC_1_3",
                     "PC1_3_with_clin",
                     "CL_UNCORR"]  # Modify as needed.
SELECTED_MODELS = ["XGBoost", "RandomForest", "MLP", "ElasticNet"]
SELECTED_PREDICTORS = ["ER", "PR", "HER2"]  # Modify as needed.
# ------------------------------------------

# Reference AUC values (update as needed)
REF_AUC = {
    "ER": 0.65,
    "PR": 0.60,
    "HER2": 0.50
}

# File paths
BASE_DIR = Path.cwd() / "data" / "final" / "lucy"
CSV_INPUT = BASE_DIR / "summary_auc.csv"         # Input CSV file path (same format as summary_auc.csv)
FIGURES_DIR = BASE_DIR / "figures" / "custom"      # Folder for custom figures
CSV_OUT = BASE_DIR / "summary_filtered.csv"        # Output filtered CSV

# Ensure output directories exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Read the CSV file
df = pd.read_csv(CSV_INPUT)

# Reshape DataFrame: convert wide format (one column per model) to long format
models = ["ElasticNet", "LASSO", "LogisticRegression", "MLP", "RandomForest", "SVM", "XGBoost"]
df_melt = pd.melt(df, id_vars=["predictor", "dataset"], value_vars=models,
                  var_name="model", value_name="auc")

# Filter the DataFrame based on user-selected datasets, models, and predictors
df_filtered = df_melt[
    df_melt["dataset"].isin(SELECTED_DATASETS) &
    df_melt["model"].isin(SELECTED_MODELS) &
    df_melt["predictor"].isin(SELECTED_PREDICTORS)
].copy()

# Write the filtered CSV summary
df_filtered.to_csv(CSV_OUT, index=False)
print(f"Saved filtered summary CSV: {CSV_OUT}")

# Create a combined column for model and predictor (e.g. "XGBoost ER")
df_filtered['model_predictor'] = df_filtered['model'] + " " + df_filtered['predictor']

# --------------------
# Grouped bar chart for AUC by dataset (for each predictor)
for predictor in df_filtered["predictor"].unique():
    sub_df = df_filtered[df_filtered["predictor"] == predictor]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sub_df, x="dataset", y="auc", hue="model", palette="Set2")
    # Set the y-axis lower bound to 0.45
    plt.ylim(0.45, ax.get_ylim()[1])
    plt.title(f"{predictor}: AUC by Dataset and Model")
    plt.ylabel("AUC")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45)
    # Add reference line if available
    if predictor in REF_AUC:
        plt.axhline(REF_AUC[predictor], color='red', linestyle='--', label='Reference AUC')
        plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    barplot_path = FIGURES_DIR / f"{predictor}_auc_barplot.png"
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved figure: {barplot_path}")

# --------------------
# Grouped bar chart for AUC by dataset (for each model)
for model in df_filtered["model"].unique():
    sub_df = df_filtered[df_filtered["model"] == model]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sub_df, x="dataset", y="auc", hue="predictor", palette="Set1")
    # Set the y-axis lower bound to 0.45
    plt.ylim(0.45, ax.get_ylim()[1])
    plt.title(f"{model}: AUC by Dataset and Predictor")
    plt.ylabel("AUC")
    plt.xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    barplot_path = FIGURES_DIR / f"{model}_auc_barplot.png"
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved figure: {barplot_path}")

for predictor in df_filtered["predictor"].unique():
    sub_df = df_filtered[df_filtered["predictor"] == predictor]
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=sub_df, x="model", y="auc", hue="dataset", palette="Set3")
    # Set the y-axis lower bound to 0.45
    plt.ylim(0.45, ax.get_ylim()[1])
    plt.title(f"{predictor}: AUC by Model and Dataset")
    plt.ylabel("AUC")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    # Add reference line if available
    if predictor in REF_AUC:
        plt.axhline(REF_AUC[predictor], color='red', linestyle='--', label='Reference AUC')
        plt.legend(title="Dataset", loc="lower right")
    plt.tight_layout()
    barplot_path = FIGURES_DIR / f"{predictor}_auc_barplot_by_model.png"
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved figure: {barplot_path}")

print("All figures generated successfully.")