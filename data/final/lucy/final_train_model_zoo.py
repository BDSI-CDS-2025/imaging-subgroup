'''
final_train_model_zoo.py
Trains four models (Random Forest, XGBoost, SVM, MLP) on every dataset
in data/final to predict each of the three targets (ER, PR, HER2).

Usage:
    python train_model_zoo_binary.py

Flags:
    Set TARGETS, DATASETS, MODELS to control which to train.
'''

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.svm import SVC
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# === FLAGS ===
TARGETS = {'ER': True, 'PR': True, 'HER2': True}  # Set to False to skip
MODELS = {'RandomForest': True, 'XGBoost': True, 'MLP': True, 'SVM': True}  # Set to False to skip

DATA_DIR = Path.cwd() / "data" / "final"
CLINICAL_PATH = Path.cwd() / "data" / "raw" / "clinicalData_clean.csv"
RESULTS_DIR = Path.cwd() / "data" / "final" / "lucy" / "results"

N_SPLITS = 5

# List all CSVs in data/final
DATASETS = {'VARS_IN_PC1': False,
            'CL_UNCORR': False,
            'VARS_IN_PC_90': False,
            'VARS_IN_PC_90_with_clin': True,
            'PC_90': True,
            'VARS_IN_PC_1_3_with_clin': True,
            'VARS_IN_PC_1_3': True,
            'ALL_IMG': True,
            'PC1': True,
            'PC1_3': True,
            'PC1_3_with_clin': True,
            'ALL_IMG_with_clin': True,
            'CL_UNCORR_with_clin': True,
            'VARS_IN_PC1_with_clin': True,
            'PC_90_with_clin': True,
            'PC1_with_clin': True}

rf_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': Categorical(['balanced', None])
}
xgb_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [1, 5, 10]
}
mlp_grid = {
    'hidden_layer_sizes': Integer(50, 100),
    'activation': Categorical(['relu', 'tanh']),
    'solver': Categorical(['adam', 'sgd']),
    'alpha': Real(0.0001, 0.01, prior='log-uniform'),
    'learning_rate_init': Real(0.001, 0.01, prior='log-uniform')
}
svm_grid = {
    'C': Real(1e-3, 1e3, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf']),
    'gamma': Real(1e-4, 1e-1, prior='log-uniform')
}

def load_data(X_path, target):
    features = pd.read_csv(X_path)
    features.rename(columns={'Patient.ID': 'Patient ID'}, inplace=True, errors='ignore')
    clin = pd.read_csv(CLINICAL_PATH)
    data = features.merge(clin[['Patient ID', target]], on='Patient ID', how='inner')
    data = data.drop('Unnamed: 0', axis=1, errors='ignore').dropna()
    y = data[target]
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = data.drop([target, 'Patient ID'], axis=1, errors='ignore')

    # Treat specified columns as categorical if present
    categorical_cols = ["Menopause (at diagnosis)", "Race and Ethnicity", "Staging(Tumor Size)# [T]"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')

    # Sanitize column names for XGBoost (remove [, ], <, and >)
    X.columns = X.columns.str.replace(r'[\[\]<>]', '', regex=True)
    
    return X, y, le

def plot_and_save_roc(y_true, y_score, le, save_dir):
    plt.figure(figsize=(7, 5))
    if len(le.classes_) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        auc = roc_auc_score(y_true, y_score[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    else:
        auc = roc_auc_score(y_true, y_score, multi_class='ovr')
        for i in range(len(le.classes_)):
            fpr, tpr, _ = roc_curve(y_true == i, y_score[:, i])
            plt.plot(fpr, tpr, label=f'Class {le.inverse_transform([i])[0]} (AUC = {roc_auc_score(y_true == i, y_score[:, i]):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "roc.png")
    plt.close()
    return auc

def plot_and_save_confusion_matrix(cm, le, save_dir):
    plt.figure(figsize=(6, 5))
    labels = le.inverse_transform(range(cm.shape[0]))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / "confusion.png")
    plt.close()

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def train_and_save(model_name, model, param_grid, X, y, le, save_dir):
    print(f"\tTraining {model_name}...")
    if model_name == "SVM":
        # SVM: no BayesSearchCV, just default params
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        y_pred = np.zeros_like(y)
        y_proba = np.zeros((len(y), len(le.classes_)))
        for i, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, y), total=3, desc="CV Folds")):
            model.fit(X.iloc[train_idx], y[train_idx])
            y_pred[test_idx] = model.predict(X.iloc[test_idx])
            y_proba[test_idx] = model.predict_proba(X.iloc[test_idx])
        best_params = model.get_params()
    else:
        # XGBoost needs categorical columns- the estimator is being cloned
        # in BayesSearchCV or crass_val_predict so they types are not
        # handled as expected
        if model_name == "XGBoost":
            # Make a copy of X and convert any categorical columns to integer codes
            X_fit = X.copy()
            cat_cols = X_fit.select_dtypes(include=["category"]).columns
            for col in cat_cols:
                X_fit[col] = X_fit[col].cat.codes
        else:
            X_fit = X
        
        search = BayesSearchCV(
            estimator=model,
            search_spaces=param_grid,
            n_iter=32,
            cv=N_SPLITS,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        # For XGBoost, use X_fit (with categorical columns as codes); otherwise, use X.
        fit_X = X_fit if model_name == "XGBoost" else X
        search.fit(fit_X, y)
        best_params = search.best_params_
        model = model.__class__(**best_params)
        y_pred = cross_val_predict(model, fit_X, y, cv=N_SPLITS)
        y_proba = cross_val_predict(model, fit_X, y, cv=N_SPLITS, method='predict_proba')
        # Save hyperparameters search results
        save_json({
            "best_params": best_params,
            "best_score": search.best_score_,
            "cv_results": {
                "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
                "std_test_score": search.cv_results_["std_test_score"].tolist(),
                "params": search.cv_results_["params"]
            }
        }, save_dir / "hyperparams.json")
    acc = accuracy_score(y, y_pred)
    auc = plot_and_save_roc(y, y_proba, le, save_dir)
    cm = confusion_matrix(y, y_pred)
    plot_and_save_confusion_matrix(cm, le, save_dir)
    metrics = {
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "best_params": best_params
    }
    save_json(metrics, save_dir / "metrics.json")
    joblib.dump(model, save_dir / "model.joblib")
    print(f"\tSaved {model_name} model, metrics, and plots to {save_dir}")
    return metrics

def main():
    for target, target_flag in TARGETS.items():
        if not target_flag:
            continue
        print(f"\n=== Target: {target} ===")
        for dataset, dataset_flag in DATASETS.items():
            if not dataset_flag:
                continue
            print(f"\nDataset: {dataset}")
            X_path = DATA_DIR / f"{dataset}.csv"
            X, y, le = load_data(X_path, target)
            for model_name, model_flag in MODELS.items():
                if not model_flag:
                    continue
                model_dir = RESULTS_DIR / target / dataset / model_name
                os.makedirs(model_dir, exist_ok=True)
                if model_name == "RandomForest":
                    model = RandomForestClassifier()
                    param_grid = rf_grid
                elif model_name == "XGBoost":
                    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', enable_categorical=True)
                    param_grid = xgb_grid
                elif model_name == "MLP":
                    model = MLPClassifier(max_iter=2000)
                    param_grid = mlp_grid
                elif model_name == "SVM":
                    model = SVC(probability=True, kernel='rbf', C=1.0, gamma='scale', random_state=42)
                    param_grid = svm_grid
                else:
                    continue
                train_and_save(model_name, model, param_grid, X, y, le, model_dir)
            print(f"Completed all models for {target} - {dataset}")
        print(f"Completed all datasets for target {target}")
    print("All training complete.")

if __name__ == "__main__":
    main()