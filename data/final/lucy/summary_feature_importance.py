import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError

# --- Configuration Dictionary ---
# Toggle folders via the "enabled" flag and update "path" as needed.
CONFIG = {
    "DATA_DIR": {
        "enabled": True,
        "path": Path.cwd() / "data" / "final"  # CSV files, e.g., PC_90.csv
    },
    "RESULTS_DIR": {
        "enabled": True,
        "path": Path.cwd() / "data" / "final" / "lucy" / "results"  # e.g., results/ER/PC_90/model.joblib
    },
    "CLINICAL_PATH": {
        "enabled": True,
        "path": Path.cwd() / "data" / "raw" / "clinicalData_clean.csv"
    }
}

def load_data(X_path, target, predictors=None):
    """
    Loads the feature CSV and clinical data.
    If predictors (i.e., a list of column names) is provided, only those columns (plus "Patient ID")
    are loaded as features—otherwise the full CSV is read.
    """
    features = pd.read_csv(X_path)
    features.rename(columns={'Patient.ID': 'Patient ID'}, inplace=True, errors='ignore')
    if predictors:
        required_cols = ['Patient ID'] + predictors
        features = features[required_cols]
    clin = pd.read_csv(CONFIG["CLINICAL_PATH"]["path"])
    data = features.merge(clin[['Patient ID', target]], on='Patient ID', how='inner')
    data = data.drop('Unnamed: 0', axis=1, errors='ignore').dropna()
    y = data[target]
    X = data.drop([target, 'Patient ID'], axis=1, errors='ignore')
    return X, y

def compute_permutation_importance(model, X, y, scoring='accuracy', n_repeats=10, random_state=42):
    """
    Computes permutation importance for a given model.
    
    For models that are missing the classes_ and n_classes_ attributes,
    we assume a binary outcome and manually set:
        classes_ = np.array([0, 1])
        n_classes_ = 2
    Otherwise, if classes_ exists but n_classes_ is missing, we try a dummy prediction.
    """
    # If classes_ attribute is missing, assume binary classification.
    if not hasattr(model, "classes_"):
        model.__dict__["classes_"] = np.array([0, 1])
    # If n_classes_ is missing, assign it
    if not hasattr(model, "n_classes_"):
        try:
            # Try a dummy prediction to force initialization (useful for some models)
            _ = model.predict(X.head(1))
        except Exception as e:
            print("Error during dummy prediction:", e)
            # If dummy prediction fails, we assume binary classification.
        model.__dict__["n_classes_"] = len(model.__dict__["classes_"])
    
    try:
        result = permutation_importance(
            model, X, y, scoring=scoring,
            n_repeats=n_repeats, random_state=random_state, n_jobs=-1
        )
    except NotFittedError as nfe:
        print("NotFittedError: Model is not fitted. Skipping this model.")
        return None
    except Exception as e:
        print("Error computing permutation importance:", e)
        return None
    
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    imp_df.sort_values(by='importance_mean', ascending=False, inplace=True)
    return imp_df

def analyze_feature_importance(target='ER', selected_datasets=None, predictors=None):
    records = []
    
    if not CONFIG["RESULTS_DIR"]["enabled"]:
        print("RESULTS_DIR is disabled in the configuration.")
        return None, None
    
    target_results_dir = CONFIG["RESULTS_DIR"]["path"] / target
    if not target_results_dir.exists():
        print(f"No results directory exists for target '{target}'. Please ensure {target_results_dir} exists.")
        return None, None

    available_dirs = [d for d in target_results_dir.iterdir() if d.is_dir()]
    if selected_datasets:
        available_dirs = [d for d in available_dirs if d.name in selected_datasets]
    if not available_dirs:
        print(f"No matching dataset directories found for target '{target}'.")
        return None, None

    for dataset_dir in available_dirs:
        dataset = dataset_dir.name
        for model_dir in dataset_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                model_path = model_dir / "model.joblib"
                if not model_path.exists():
                    continue
                model = joblib.load(model_path)
                
                if not CONFIG["DATA_DIR"]["enabled"]:
                    print("DATA_DIR is disabled in the configuration.")
                    continue
                X_path = CONFIG["DATA_DIR"]["path"] / f"{dataset}.csv"
                try:
                    X, y = load_data(X_path, target, predictors)
                except Exception as e:
                    print(f"Error loading data for dataset {dataset}: {e}")
                    continue
                print(f"Computing importance for {target} - {dataset} - {model_name}")
                imp_df = compute_permutation_importance(model, X, y)
                if imp_df is None:
                    continue
                imp_df['dataset'] = dataset
                imp_df['model'] = model_name
                records.append(imp_df)

    if records:
        all_imp = pd.concat(records, ignore_index=True)
        summary = all_imp.groupby('feature')['importance_mean'].mean().reset_index()
        summary = summary.sort_values(by='importance_mean', ascending=False)
        return summary, all_imp
    else:
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Feature Importance Analysis")
    parser.add_argument("--target", type=str, default="PR", help="Target variable (e.g. ER, PR, HER2)")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of dataset names to include")
    parser.add_argument("--predictors", type=str, default=None, help="Comma-separated list of predictor columns to include")
    args = parser.parse_args()
    
    target = args.target
    selected_datasets = args.datasets.split(",") if args.datasets else None
    predictors = args.predictors.split(",") if args.predictors else None
    
    summary, all_imp = analyze_feature_importance(target, selected_datasets, predictors)
    
    if summary is not None:
        print("Average Feature Importance across models and datasets:")
        print(summary.head(10))
        plt.figure(figsize=(10, 6))
        sns.barplot(data=summary.head(10), x='importance_mean', y='feature')
        plt.xlabel('Average Importance')
        plt.title(f'Top 10 Important Features for {target}')
        plt.tight_layout()
        output_path = Path.cwd() / f"feature_importance_{target}.png"
        plt.savefig(output_path)
        plt.show()
    else:
        print("No feature importance records found.")

if __name__ == "__main__":
    main()