'''
visualize_model_zoo_binary.py
Produces figures demonstrating the relative performance
of all models across all datasets.
'''

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

path = Path.cwd()
print(path)
BASE_DIR = path / "scripts" / "python" / "all_model_results"
SAVE_DIR = path / "results" / "figures" / "binary-model-zoo"
FEATURES = ["ER", "PR", "HER2"]  # Add "MOL-SUBTYPE" if you want multiclass
DATASETS = ["ALL_IMG", "PC1", "PC1_3", "PC_90", "VARS_IN_PC1", "VARS_IN_PC_1_3", "VARS_IN_PC_90"]
REF_AUC = {'ER' : .649, 'PR' : .622, 'HER2' : .5}

def collect_metrics(base_dir=BASE_DIR, features=FEATURES, datasets=DATASETS):
    records = []
    for feature in features:
        feature_dir = os.path.join(base_dir, feature)
        for dataset in datasets:
            dataset_dir = os.path.join(feature_dir, dataset)
            metrics_path = os.path.join(dataset_dir, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                for model, vals in metrics.items():
                    records.append({
                        "feature": feature,
                        "dataset": dataset,
                        "model": model,
                        "accuracy": vals.get("accuracy"),
                        "auc": vals.get("auc")
                    })
    # Ensure DataFrame always has the expected columns
    return pd.DataFrame(records, columns=["feature", "dataset", "model", "accuracy", "auc"])

def plot_grouped_bar(df, metric="accuracy"):
    if df.empty:
        print("No metrics found. DataFrame is empty.")
        return
    for feature in df["feature"].unique():
        plt.figure(figsize=(10, 6))
        sub = df[df["feature"] == feature]
        sns.barplot(
            data=sub,
            x="dataset", y=metric, hue="model",
            palette="Set2"
        )
        plt.title(f"{feature}: Model {metric.capitalize()} by Dataset")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend(title="Model")
        # Plot horizontal reference line
        if metric == "auc" and feature in REF_AUC:
            plt.axhline(REF_AUC[feature], color='red', linestyle='--', label='Reference AUC')
            plt.legend(title="Model", loc="lower right")
        DEST = SAVE_DIR / (feature +'-auc' + '.png')
        plt.savefig(DEST)

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

if __name__ == "__main__":
    df = collect_metrics()
    plot_grouped_bar(df, metric="auc")  # or metric="auc"
    plot_overall_model_performance(df, metric="auc")  # or metric="auc"