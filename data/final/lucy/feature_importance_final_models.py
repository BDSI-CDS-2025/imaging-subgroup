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
import joblib # to load in models
from sklearn.inspection import permutation_importance # for interpretabillity

RES_DIR = Path.cwd() / "data" / "final" / "lucy" / "importance" / "best_models"
CLIN_DATA_PATH = Path.cwd() / "data" / "raw" / "clinicalData_clean.csv"

MODEL_INFO = [
    {'model' : Path.cwd() / "data" / "final" / "lucy" / "results/ER/CL_UNCORR_with_clin/XGBoost/model.joblib", # ER
     'target' : 'ER',
     'train' : Path.cwd() / "data" / "final" / "CL_UNCORR_with_clin.csv"},
    {'model' : Path.cwd() / "data" / "final" / "lucy" / "results/PR/ALL_IMG/RandomForest/model.joblib", # PR
     'target' : 'PR',
     'train' : Path.cwd() / "data" / "final" / "ALL_IMG.csv"},
    {'model' : Path.cwd() / "data" / "final" / "lucy" / "results/HER2/PC1_with_clin/LogisticRegression/model.joblib", # HER2
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
    print("Columns in loaded data:", list(data.columns))

    # Rename column if it doesn't match training
    # Adjust this mapping as needed to match your training data feature names.
    data.rename(columns={"Staging(Tumor Size)# [T]": "Staging(Tumor Size)# T"}, inplace=True)

    y = data[target]
    X = data.drop([target, 'Patient ID'], axis=1, errors='ignore')
    return X, y

for m in MODEL_INFO:
    model = joblib.load(m['model'])
    X, y = load_data(m['train'], m['target'])
    # Fix for XGBClassifier: add n_classes_ attribute if missing
    if not hasattr(model, 'n_classes_'):
        model.n_classes_ = len(np.unique(y))

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