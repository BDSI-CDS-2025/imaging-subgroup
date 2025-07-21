'''
feature_importance_final_models.py
Performs feature importance analysis on the following three models
ER: XGBoost on Uncorrelated + Clinical
PR: Random Forest on All Imaging
HER2: Logistic Regression on PC1 + Clinical
'''

from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib # to load in models
from sklearn.inspection import permutation_importance # for interpretabillity
from sklearn.model_selection import train_test_split

# Each of the models that will be retrained
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

RES_DIR = Path.cwd() / "data" / "final" / "lucy" / "importance" / "best_models"
CLIN_DATA_PATH = Path.cwd() / "data" / "raw" / "clinicalData_clean.csv"

MODEL_INFO = [
    {'model_params' : Path.cwd() / "data" / "final" / "lucy" / "results/ER/CL_UNCORR_with_clin/XGBoost/hyperparams.json", # ER
     'target' : 'ER',
     'train' : Path.cwd() / "data" / "final" / "CL_UNCORR_with_clin.csv"},
    {'model_params' : Path.cwd() / "data" / "final" / "lucy" / "results/PR/ALL_IMG/RandomForest/hyperparams.json", # PR
     'target' : 'PR',
     'train' : Path.cwd() / "data" / "final" / "ALL_IMG.csv"},
    {'model_params' : Path.cwd() / "data" / "final" / "lucy" / "results/HER2/PC1_with_clin/LogisticRegression/hyperparams.json", # HER2
     'target' : 'HER2',
     'train' : Path.cwd() / "data" / "final" / "PC1_with_clin.csv"}
]

def load_data(X_path, target):
    """
    Loads the feature CSV and clinical data.
    If predictors (i.e., a list of column names) is provided, only those columns (plus "Patient ID")
    are loaded as features—otherwise the full CSV is read.
    """
    features = pd.read_csv(X_path)
    features.rename(columns={'Patient.ID': 'Patient ID'}, inplace=True, errors='ignore')
    clin = pd.read_csv(CLIN_DATA_PATH)
    data = features.merge(clin[['Patient ID', target]], on='Patient ID', how='inner')
    data = data.drop('Unnamed: 0', axis=1, errors='ignore').dropna()
    
    # Debug print: show feature names used in prediction
    # print("Columns in loaded data:", list(data.columns))

    # Rename column if it doesn't match training
    # Adjust this mapping as needed to match your training data feature names.
    data.rename(columns={"Staging(Tumor Size)# [T]": "Staging(Tumor Size)# T"}, inplace=True)

    y = data[target]
    X = data.drop([target, 'Patient ID'], axis=1, errors='ignore')
    return X, y

for m in MODEL_INFO:
    print(f'----- Running feature importance for {m["target"]}')
    # Load hyperparameters
    with open(m['model_params'], 'r') as f:
        params = json.load(f)
    X, y = load_data(m['train'], m['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select model type based on target/model_params path
    if 'XGBoost' in str(m['model_params']):
        model = XGBClassifier(**params["best_params"])
    elif 'RandomForest' in str(m['model_params']):
        model = RandomForestClassifier(**params["best_params"])
    elif 'LogisticRegression' in str(m['model_params']):
        model = LogisticRegression(**params["best_params"], max_iter=1000)
    else:
        raise ValueError(f"Unknown model type for {m['model_params']}")
    
    print(f'\t----- Training')
    # Train model
    model.fit(X_train, y_train)

    # Save model
    model_path = RES_DIR / f"{m['target']}_best_model.joblib"
    joblib.dump(model, model_path)

    result = permutation_importance(
            model, X, y, scoring='accuracy',
            n_repeats=10, random_state=3, n_jobs=-1
            )
    imp_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    imp_df.sort_values(by='importance_mean', ascending=False, inplace=True)
    imp_df.to_csv(RES_DIR / (m['target'] + '_best_model_feature_importance.csv'))
    print(f'----- Results saved to {RES_DIR / (m["target"] + "_best_model_feature_importance.csv")}')