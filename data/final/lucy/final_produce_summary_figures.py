'''
final_produce_summary_figures.py
Takes all of the logged statistics in the subfolders and produces
figures that summarize model performance by dataset and by model.

Produces summary .csv as well.
'''

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

path = Path.cwd()
# Updated base directory for results
BASE_DIR = path / "data" / "final" / "lucy" / "results"
SAVE_DIR = path / "data" / "final" / "lucy" / "figures" / "binary-model-zoo"
CSV_OUT = path / "data" / "final" / "lucy" / "summary_auc.csv"
# Expected targets (predictors)
PREDICTORS = ["ER", "PR", "HER2"]
# List of datasets you want to include
DATASETS = ["ALL_IMG", "CL_UNCORR", "PC1", "PC1_3", "PC_90", "VARS_IN_PC1", "VARS_IN_PC_1_3", "VARS_IN_PC_90"]
REF_AUC = {'ER' : .649, 'PR' : .622, 'HER2' : .5}

def collect_metrics(base_dir=BASE_DIR):
    records = []
    for predictor in PREDICTORS:
        predictor_dir = base_dir / predictor
        if not predictor_dir.exists():
            continue
        # Iterate over datasets within each predictor folder
        for dataset in os.listdir(predictor_dir):
            dataset_dir = predictor_dir / dataset
            if not dataset_dir.is_dir():
                continue
            # Iterate over models in the dataset folder
            for model in os.listdir(dataset_dir):
                model_dir = dataset_dir / model
                metrics_path = model_dir / "metrics.json"
                if metrics_path.exists():
                    with open(metrics_path) as f:
                        metrics = json.load(f)
                    records.append({
                        "predictor": predictor,
                        "dataset": dataset,
                        "model": model,
                        "accuracy": metrics.get("accuracy"),
                        "auc": metrics.get("auc")
                    })
    # Create a DataFrame with the expected columns
    return pd.DataFrame(records, columns=["predictor", "dataset", "model", "accuracy", "auc"])

def plot_grouped_bar(df, metric="accuracy"):
    if df.empty:
        print("No metrics found. DataFrame is empty.")
        return
    for predictor in df["predictor"].unique():
        plt.figure(figsize=(10, 6))
        sub = df[df["predictor"] == predictor]
        sns.barplot(
            data=sub,
            x="dataset", y=metric, hue="model",
            palette="Set2"
        )
        plt.title(f"{predictor}: Model {metric.capitalize()} by Dataset")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title="Model")
        # Plot horizontal reference line for AUC, if applicable
        if metric == "auc" and predictor in REF_AUC:
            plt.axhline(REF_AUC[predictor], color='red', linestyle='--', label='Reference AUC')
            plt.legend(title="Model", loc="lower right")
        outpath = SAVE_DIR / f"{predictor}-{metric}.png"
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(outpath, dpi=300)
        plt.close()
        print(f"Saved: {outpath}")

def plot_dataset_performance_by_feature(df, metric="accuracy"):
    if df.empty:
        print("No metrics found. DataFrame is empty.")
        return
    plt.figure(figsize=(12, 6))
    marker_dict = {"ER": "o", "PR": "s", "HER2": "D"}
    for predictor, marker in marker_dict.items():
        sub = df[df["predictor"] == predictor]
        sns.stripplot(
            data=sub,
            x="dataset",
            y=metric,
            hue="model",
            dodge=True,
            jitter=True,
            palette="Set2",
            size=8,
            alpha=0.8,
            marker=marker
        )
    plt.title(f"{metric.capitalize()} by Dataset, Predictor and Model")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    from matplotlib.lines import Line2D
    feature_handles = [
        Line2D([0], [0], marker=marker_dict[f], color='w', label=f, markerfacecolor='gray', markersize=10)
        for f in marker_dict
    ]
    plt.legend(handles=handles[:len(df["model"].unique())] + feature_handles,
               labels=labels[:len(df["model"].unique())] + list(marker_dict.keys()),
               bbox_to_anchor=(1.05, 1), loc='upper left', title="Model / Predictor")
    outpath = SAVE_DIR / f"{metric}_by_dataset_predictor_model_shape.png"
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

def plot_overall_model_performance(df, metric="accuracy"):
    if df.empty:
        print("No metrics found. DataFrame is empty.")
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="model", y=metric, palette="Set2", hue="model")
    sns.swarmplot(data=df, x="model", y=metric, color=".25", alpha=0.7)
    plt.title(f"Overall {metric.capitalize()} Distribution by Model")
    plt.ylabel(metric.capitalize())
    plt.xlabel("Model")
    plt.tight_layout()
    outpath = SAVE_DIR / f"overall_{metric}_by_model.png"
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"Saved: {outpath}")

def create_summary_csv(df, csv_out=CSV_OUT):
    # Pivot the DataFrame so that for each predictor and dataset, each model's AUC is in its own column.
    if df.empty:
        print("No metrics found. DataFrame is empty.")
        return
    pivot_df = df.pivot_table(index=["predictor", "dataset"], columns="model", values="auc").reset_index()
    pivot_df.columns.name = None  # remove pivot table grouping name if present
    pivot_df.to_csv(csv_out, index=False)
    print(f"Saved summary CSV: {csv_out}")

if __name__ == "__main__":
    df = collect_metrics()
    if df.empty:
        print("No metrics found. Exiting.")
    else:
        plot_grouped_bar(df, metric="auc")
        plot_overall_model_performance(df, metric="auc")
        plot_dataset_performance_by_feature(df, metric="auc")
        create_summary_csv(df)