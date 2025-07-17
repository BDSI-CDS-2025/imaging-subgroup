'''
feature_importance_final_models.py
Performs feature importance analysis on the following three models
ER: XGBoost on Uncorrelated + Clinical
PR: Random Forest on All Imaging
HER2: Logistic Regression on PC1 + Clinical
'''

from pathlib import Path
import pandas as pd

CLIN_DATA_PATH = Path.cwd() / "data" / "raw" / "clinicalData_clean.csv"

ER_MODEL = "results/ER/CL_UNCORR_with_clin/XGBoost/model.joblib"
PR_MODEL = "results/PR/ALL_IMG/RandomForest/model.joblib"
HER2_MODEL = "results/HER2/PC1_with_clin/model.joblib"

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