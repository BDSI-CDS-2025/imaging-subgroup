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
import matplotlib.colors as mcolors

# ------------------------------------------
# User adjustable parameters - modify these as needed
SELECTED_DATASETS = ['VARS_IN_PC1',
            'CL_UNCORR',
            'VARS_IN_PC_90',
            'VARS_IN_PC_90_with_clin',
            'PC_90',
            'VARS_IN_PC_1_3_with_clin',
            'VARS_IN_PC_1_3',
            'ALL_IMG',
            'PC1',
            'PC1_3',
            'PC1_3_with_clin',
            'ALL_IMG_with_clin',
            'CL_UNCORR_with_clin',
            'VARS_IN_PC1_with_clin',
            'PC_90_with_clin',
            'PC1_with_clin']
BARPLOT_DATASETS = ['ALL_IMG', 'ALL_IMG_with_clin']
SELECTED_MODELS = ["XGBoost", "RandomForest", "MLP", "LogisticRegression", "Superlearner", "GMM" ,"K-Means", "SVM", "LASSO", "ElasticNet"]
SELECTED_PREDICTORS = ["ER", "PR", "HER2"]  # Modify as needed.
# ------------------------------------------

# Reference AUC values (update as needed)
REF_AUC = {
    "ER": 0.649,
    "PR": 0.622,
    "HER2": 0.50
}

# FIXING THE COLUMN NAMES
CUSTOM_DATASET_LABELS = {
    "ALL_IMG": "All Imaging Data",
    "ALL_IMG_with_clin": "All Imaging\n+ Pre-Biopsy Clinical",

    "PC1_3": "Top Three\n PCs Per\nFeature Group",
    "VARS_IN_PC_1_3": "Raw Data Values\nfrom Top Three Covariates\nin PC1",
    "PC1_3_with_clin": "Top Three PCs\ns Per Feature Group\n+ Clinical",
    "VARS_IN_PC_1_3_with_clin":"Raw Data Values\nfrom Top Three Covariates\nin PC1\n+ Clinical",

    "PC1": "Top PC\ Per Feature\n Group",
    "VARS_IN_PC1" : "Raw Data Values\nfrom Top\n Covaraiates in PC1",
    "PC1_with_clin": "Top PC Per\nFeature Group\n+ Clinical",
    "VARS_IN_PC1_with_clin" : "Raw Data Values\nfrom Top\n Covaraiate in\n PC1 + Clinical",

    "PC_90": "Enough PCs\nto Explain 90%\nVariance per\nFeature Group",
    "VARS_IN_PC_90": "Raw Data Values\nfrom Enough PCs to\nExplain 90% of\nVariance per\nFeature Group",
    "PC_90_with_clin": "Enough PCs to\nExplain 90% of\nVariance per\nFeature Group\n+ Clinical",
    "VARS_IN_PC_90_with_clin": "Raw Data Values\nfrom PCs to\nExplain 90% of\nVariance per\nFeature Group\n+ Clinical",

    "CL_UNCORR": "Hierarchical Clustering",
    "CL_UNCORR_with_clin": "Hierarchical Clustering\n+ Clinical"
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
models = ["ElasticNet", "LASSO", "LogisticRegression", "MLP", "RandomForest", "SVM", "XGBoost", "Superlearner", "Clustering", "GMM" ,"K-Means"]
df_melt = pd.melt(df, id_vars=["predictor", "dataset"], value_vars=models,
                  var_name="model", value_name="auc")

# Filter the DataFrame based on user-selected datasets, models, and predictors
df_filtered = df_melt[
    df_melt["dataset"].isin(SELECTED_DATASETS) &
    df_melt["model"].isin(SELECTED_MODELS) &
    df_melt["predictor"].isin(SELECTED_PREDICTORS)
].copy()

df_barplot = df_melt[df_melt["dataset"].isin(BARPLOT_DATASETS)].copy()

# Write the filtered CSV summary
df_filtered.to_csv(CSV_OUT, index=False)
print(f"Saved filtered summary CSV: {CSV_OUT}")

# Create a combined column for model and predictor (e.g. "XGBoost ER")
df_filtered['model_predictor'] = df_filtered['model'] + " " + df_filtered['predictor']


# Barchart for AUC by model
for predictor in df_barplot["predictor"].unique():
    sub_df = df_barplot[df_barplot["predictor"] == predictor]
    plt.figure(figsize=(10, 6))
    reference = REF_AUC.get(predictor, 0)
    sub_df['auc_diff'] = sub_df['auc'] - reference

    # Set3 and Set2
    ax = sns.barplot(data=sub_df, x="model", y="auc_diff", hue="dataset", palette="Set2")

    plt.title(f"{predictor}: AUC by Model and Dataset")
    plt.ylabel("Difference from Reference AUC")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    
    plt.axhline(0, color='red', linestyle='--', label='_nolegend_')

    # Clear the existing annotations and use a different approach
    for p in ax.containers:
        # This avoids the ghost annotations issue
        ax.bar_label(p, labels=[f"{h + reference:.4f}" for h in p.datavalues], 
                    padding=3, fontsize=5)

    # Fix legend
    handles, labels = ax.get_legend_handles_labels()
    new_labels = [CUSTOM_DATASET_LABELS.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, new_labels, title="Dataset", 
          bbox_to_anchor=(1.05, 1), 
          loc='upper left', 
          borderaxespad=0.)

    plt.tight_layout()
    barplot_path = FIGURES_DIR / f"{predictor}_auc_barplot_by_model.png"
    plt.savefig(barplot_path, dpi=300)
    plt.close()
    print(f"Saved figure: {barplot_path}")

# Boxplot for AUC by Model for each Predictor
for predictor in df_filtered["predictor"].unique():
    sub_df = df_filtered[df_filtered["predictor"] == predictor]
    plt.figure(figsize=(10, 6))

    ax = sns.boxplot(data=sub_df, x="model", y="auc", palette="Set3", showfliers=False)

    # Overlay point-data on top
    sns.stripplot(x="model", y="auc", data=sub_df, color="black", size=4, jitter=True)
    
    # Add a horizontal line for the reference AUC value if available
    reference = REF_AUC.get(predictor, None)
    if reference is not None:
        plt.axhline(reference, color='red', linestyle='--', label="Reference AUC")
    
    plt.title(f"{predictor}: AUC Boxplot by Model")
    plt.ylabel("AUC")
    plt.xlabel("Model")
    plt.xticks(rotation=45)
    
    if reference is not None:
        plt.legend(loc="upper right")
    
    plt.tight_layout()
    boxplot_path = FIGURES_DIR / f"{predictor}_auc_boxplot.png"
    plt.savefig(boxplot_path, dpi=300)
    plt.close()
    print(f"Saved figure: {boxplot_path}")

print("Boxplots generated successfully.")

# Heatmaps of AUC by Model and Dataset (with clustering) for each Predictor
for predictor in df_filtered["predictor"].unique():
    sub_df = df_filtered[df_filtered["predictor"] == predictor]

    # Use pivot_table with aggfunc='first' to avoid aggregation artifacts
    pivot_table = sub_df.pivot_table(index="model", columns="dataset",
                                     values="auc", aggfunc='first')
    # Fill missing values so that there are only finite numbers
    pivot_table = pivot_table.fillna(0)

    # Map datasets to numbers
    dataset_list = list(pivot_table.columns)
    dataset_numbers = list(range(1, len(dataset_list) + 1))
    dataset_map = dict(zip(dataset_list, dataset_numbers))
    pivot_table.columns = dataset_numbers

    # Set up custom colormap: .5=white, max=green
    vmin = 0.5
    vmax = pivot_table.max().max()
    if vmax <= vmin:
        vmax = vmin + 0.01  # ensure ascending order
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=(vmax+vmin)/2, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_green", ["white", "green"])
    
    plt.figure(figsize=(14.5, 10))
    ax = sns.heatmap(pivot_table, cmap=cmap, norm=norm, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={'label': 'AUC'})
    ax.set_title(f"{predictor}: AUC Heatmap by Model and Dataset", pad=20)
    
    # Update x-axis labels with custom dataset labels when available
    ax.set_xticklabels(dataset_numbers, rotation=0)

    # Create legend mapping numbers to dataset names
    legend_labels = [f"{num}: {CUSTOM_DATASET_LABELS.get(ds, ds)}" for ds, num in dataset_map.items()]
    legend_text = "\n\n".join(legend_labels)
    plt.gcf().text(1.25, 0.5, legend_text, va='center', fontsize=8, transform=ax.transAxes)

    heatmap_path = FIGURES_DIR / f"{predictor}_auc_heatmap.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

print("Heatmaps generated successfully.")

print("All figures generated successfully.")