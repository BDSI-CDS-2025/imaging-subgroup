'''
train_model_zoo_binary.py
Runs training for models when trying to predict the three
binary outcomes: HER2, ER, PR.

To run: from root imaging-subgroup
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
from sklearn.model_selection import cross_val_predict

from config import SETUP

warnings.filterwarnings("ignore")

path = Path.cwd()
RESULTS_DEST = path / "scripts" / "python" / "all_model_results"
HYPERPARAM_DEST = path / "notebooks" / "modeling"

TARGETS = ['ER', 'PR', 'HER2']

clinical_data_path = path / "data" / "raw" / "clinicalData_clean.csv"

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

def load_data(X_path, target):
    '''
    Sets up a feature matrix X and target y where the input file
    is given by X_path (can be PC features, all features, etc.) and
    y will only have the desired target to predict.
    '''
    features = pd.read_csv(X_path)
    features.rename(columns={'Patient.ID': 'Patient ID'}, inplace=True, errors='ignore')
    clin = pd.read_csv(clinical_data_path)
    data = features.merge(clin[['Patient ID', target]], on='Patient ID', how='inner')
    data = data.drop('Unnamed: 0', axis=1, errors='ignore').dropna() # Won't run if there are NA values
    y = data[target]
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))
    X = data.drop([target, 'Patient ID'], axis=1, errors='ignore')
    return X, y, le

def plot_and_save_roc(y_true, y_score, le, model_name, dataset_name, target, save_dir):
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
    plt.title(f'ROC Curve: {model_name} on {dataset_name} ({target})')
    plt.legend()
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir / f"{model_name}_roc.png")
    plt.close()
    return auc

def plot_and_save_confusion_matrix(cm, le, model_name, dataset_name, target, save_dir):
    plt.figure(figsize=(6, 5))
    labels = le.inverse_transform(range(cm.shape[0]))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name} ({target})')
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_dir / f"{model_name}_confusion.png")
    plt.close()

def log_search_results(search, model_name, all_results):
    all_results.append({
        "model": model_name,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": {
            "mean_test_score": search.cv_results_["mean_test_score"].tolist(),
            "std_test_score": search.cv_results_["std_test_score"].tolist(),
            "params": search.cv_results_["params"]
        }
    })

def main():
    for target in TARGETS:
        print(f"\n=== Processing target: {target} ===")
        for dataset_name, setup in SETUP.items():
            print(f"\nProcessing dataset: {dataset_name}")
            X, y, le = load_data(setup['X'], target)
            all_results = []
            dataset_results = {}

            # RandomForest
            rf_bayes = BayesSearchCV(
                estimator=RandomForestClassifier(),
                search_spaces=rf_grid,
                n_iter=32,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            rf_bayes.fit(X, y)
            log_search_results(rf_bayes, "RandomForest", all_results)
            rf_best_params = rf_bayes.best_params_.copy()
            #rf_best_params['class_weight'] = 'balanced'
            rf_final = RandomForestClassifier(**rf_best_params)
            y_pred = cross_val_predict(rf_final, X, y, cv=3)
            y_proba = cross_val_predict(rf_final, X, y, cv=3, method='predict_proba')
            acc = accuracy_score(y, y_pred)
            auc = plot_and_save_roc(y, y_proba, le, "RandomForest", dataset_name, target, RESULTS_DEST / target / dataset_name)
            cm = confusion_matrix(y, y_pred)
            plot_and_save_confusion_matrix(cm, le, "RandomForest", dataset_name, target, RESULTS_DEST / target / dataset_name)
            dataset_results["RandomForest"] = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix": cm.tolist(),
                "best_params": rf_best_params
            }

            # XGBoost
            xgb_bayes = BayesSearchCV(
                estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                search_spaces=xgb_grid,
                n_iter=32,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            xgb_bayes.fit(X, y)
            log_search_results(xgb_bayes, "XGBoost", all_results)
            xgb_final = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_bayes.best_params_)
            y_pred = cross_val_predict(xgb_final, X, y, cv=3)
            y_proba = cross_val_predict(xgb_final, X, y, cv=3, method='predict_proba')
            acc = accuracy_score(y, y_pred)
            auc = plot_and_save_roc(y, y_proba, le, "XGBoost", dataset_name, target, RESULTS_DEST / target / dataset_name)
            cm = confusion_matrix(y, y_pred)
            plot_and_save_confusion_matrix(cm, le, "XGBoost", dataset_name, target, RESULTS_DEST / target / dataset_name)
            dataset_results["XGBoost"] = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix": cm.tolist(),
                "best_params": xgb_bayes.best_params_
            }

            # MLP
            mlp_bayes = BayesSearchCV(
                estimator=MLPClassifier(max_iter=2000),
                search_spaces=mlp_grid,
                n_iter=32,
                cv=3,
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            mlp_bayes.fit(X, y)
            log_search_results(mlp_bayes, "MLP", all_results)
            mlp_final = MLPClassifier(max_iter=2000, **mlp_bayes.best_params_)
            y_pred = cross_val_predict(mlp_final, X, y, cv=3)
            y_proba = cross_val_predict(mlp_final, X, y, cv=3, method='predict_proba')
            acc = accuracy_score(y, y_pred)
            auc = plot_and_save_roc(y, y_proba, le, "MLP", dataset_name, target, RESULTS_DEST / target / dataset_name)
            cm = confusion_matrix(y, y_pred)
            plot_and_save_confusion_matrix(cm, le, "MLP", dataset_name, target, RESULTS_DEST / target / dataset_name)
            dataset_results["MLP"] = {
                "accuracy": acc,
                "auc": auc,
                "confusion_matrix": cm.tolist(),
                "best_params": mlp_bayes.best_params_
            }

            # Save hyperparameters for this dataset/target
            hyperparam_dir = HYPERPARAM_DEST / target
            os.makedirs(hyperparam_dir, exist_ok=True)
            with open(hyperparam_dir / f"{dataset_name}.json", "w") as f:
                json.dump(all_results, f, indent=2)

            # Save results for this dataset/target
            results_dir = RESULTS_DEST / target / dataset_name
            os.makedirs(results_dir, exist_ok=True)
            with open(results_dir / "metrics.json", "w") as f:
                json.dump(dataset_results, f, indent=2)

            print(f"Saved results and hyperparameters for {target} - {dataset_name}")

if __name__ == "__main__":
    main()