'''
train_model_zoo.py
For all three of the models (MLP, XGBoost, RF), trains the models
on each of the datasets specified in SETUP.
To run: python3 scripts/python/train_model_zoo.py 
'''

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from skopt.space import Categorical, Real, Integer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
import warnings

from config import SETUP # contains information on where to read/store information

warnings.filterwarnings("ignore") # don't show any warnings

path = Path.cwd()

TARGET = 'Mol Subtype'
RESULTS_DEST = path / "scripts" / "python" / "all_model_results"
train_patient_id_path = path / "data" / "processed" / "trainDataPatientID.csv"
clinical_data_path = path / "data" / "raw" / "clinicalData_clean.csv"

# Define search grids
rf_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'class_weight': Categorical(['balanced', 'balanced_subsample', None])
}
xgb_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'scale_pos_weight': [1, 5, 10] # tweak around the inverse-imbalance ratio
}
mlp_grid = {
    'hidden_layer_sizes': Integer(50, 100),
    'activation': Categorical(['relu', 'tanh']),
    'solver': Categorical(['adam', 'sgd']),
    'alpha': Real(0.0001, 0.01, prior='log-uniform'),
    'learning_rate_init': Real(0.001, 0.01, prior='log-uniform')
}

def load_data(X_path):
    trainPatientID = (pd.read_csv(train_patient_id_path)
                      .drop(['Unnamed: 0'], axis=1, errors='ignore')
                      .rename(columns={'Patient.ID': 'Patient ID'}, errors='ignore'))
    features = pd.read_csv(X_path)
    features.rename(columns={'Patient.ID': 'Patient ID'}, inplace=True, errors='ignore')
    features = features[features['Patient ID'].isin(trainPatientID['Patient ID'])]
    clin = pd.read_csv(clinical_data_path)
    clin = clin[clin['Patient ID'].isin(trainPatientID['Patient ID'])]
    data = features.merge(clin[['Patient ID', TARGET]], on='Patient ID', how='inner')
    data = data.drop('Unnamed: 0', axis=1, errors='ignore')
    y = data[TARGET]

    # Always treat as categorical, even if numeric
    le = LabelEncoder()
    y = le.fit_transform(y.astype(str))  # Cast to str to ensure categorical treatment

    X = data.drop([TARGET, 'Patient ID'], axis=1, errors='ignore')
    return X, y

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

def run_all():

    # Track information as training occurs

    summary_rows = []
    for key, setup in SETUP.items():
        print(f"\nProcessing dataset: {key}")
        X, y = load_data(setup['X'])
        all_results = []

        sample_w = compute_sample_weight(class_weight='balanced', y=y)

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
        rf_bayes.fit(X, y, sample_weight=sample_w)
        log_search_results(rf_bayes, "RandomForest", all_results) # switch to rf_bayes if no override
        summary_rows.append({
            "dataset": key,
            "model": "RandomForest",
            "cv_accuracy": rf_bayes.best_score_
        })

        # Hard-code that we want the tree to be balanced
        rf_best_params = rf_bayes.best_params_.copy()
        rf_best_params['class_weight'] = 'balanced'

        # Fit the final model with the overridden parameters
        rf_final = RandomForestClassifier(**rf_best_params)
        rf_final.fit(X, y, sample_weight=sample_w) # switch to rf_bayes if no override
        log_search_results(rf_final, "RandomForest-Weighted", all_results)
        summary_rows.append({
            "dataset": key,
            "model": "RandomForest-Weighted",
            "cv_accuracy": rf_final.best_score_
        })
        print("\t✅RF fit")

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
        xgb_bayes.fit(X, y, sample_weight=sample_w)
        log_search_results(xgb_bayes, "XGBoost", all_results)
        summary_rows.append({
            "dataset": key,
            "model": "XGBoost",
            "cv_accuracy": xgb_bayes.best_score_
        })
        print("\t✅XGBoost fit")

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
        mlp_bayes.fit(X, y, sample_weight=sample_w)
        log_search_results(mlp_bayes, "MLP", all_results)
        summary_rows.append({
            "dataset": key,
            "model": "MLP",
            "cv_accuracy": mlp_bayes.best_score_
        })
        print("\t✅MLP fit")

        # Save results
        with open(setup['hyperparameters_file'], "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved results to {setup['hyperparameters_file']}")
    
    summary_df = pd.DataFrame(summary_rows)
    SUMMARY_DEST = RESULTS_DEST / "train_cv_accuracy_summary.csv"
    summary_df.to_csv(SUMMARY_DEST, index = False)
    print(f'Saved cross-validated accuracy summary to {SUMMARY_DEST}')

if __name__ == "__main__":
    run_all()
