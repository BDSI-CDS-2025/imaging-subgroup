'''
mlp_pca_results_as_features.py
Runs a pca, calculating the shap values, using the PC1 coordinates as the features.
'''

import shap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

data = pd.read_csv(BASE_DIR / 'data/interim/pc_by_feature_group_for_patients.csv')
all = pd.read_csv(BASE_DIR / 'data/raw/clinicalData_clean.csv')
X = data
y = all['HER2']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)