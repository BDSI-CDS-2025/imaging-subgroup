import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data 
clinData = pd.read_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/raw/clinicalData_clean.csv') # Adjust path as necessary
imFeatures = pd.read_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/raw/imagingFeatures.csv')
# Remove 1st row of each DataFrame (as it contains extra information)
clinData = clinData.iloc[1:]
imFeatures = imFeatures.iloc[1:]

# Merge clinical and imaging data on 'Patient ID' only including 'Mol Subtype' from clinical data
fullData = pd.merge(imFeatures, clinData[['Patient ID', 'Mol Subtype']],
                    on='Patient ID', how='inner')

# Check fullData for missing values
missing_full_data = fullData.isnull().sum()
# Drop rows with missing values in fullData
fullData = fullData.dropna()
# Check dimensions after dropping missing values
print("Full Data shape after dropping NaNs:", fullData.shape)

# Define features and target variable
imFeatures = fullData.drop(columns=['Patient ID', 'Mol Subtype'])
X = imFeatures.values
y = fullData['Mol Subtype'].values
# Check shapes and NaN values in X and y
print("X shape:", X.shape)
print("y shape:", y.shape)

# Encode target labels if not already numeric
if not np.issubdtype(y.dtype, np.number):
    le = LabelEncoder()
    y = le.fit_transform(y)

# 5-fold cross-validated multinomial logistic regression
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=630)
splits = list(skf.split(X, y))

cv_accuracies = []
cv_roc_aucs = []

for fold, (train_idx, val_idx) in enumerate(splits, 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    logit_model.fit(X_train_scaled, y_train)
    y_pred = logit_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    y_proba = logit_model.predict_proba(X_val_scaled)
    roc_auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
    cv_accuracies.append(acc)
    cv_roc_aucs.append(roc_auc)
    print(f"Logistic CV Fold {fold} accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")

print(f"Mean cross-validated accuracy (5 folds): {np.mean(cv_accuracies):.4f}")
print(f"Mean cross-validated ROC-AUC (5 folds): {np.mean(cv_roc_aucs):.4f}")

# 5-fold cross-validated multinomial lasso regression
from sklearn.linear_model import LogisticRegressionCV  
lasso_cv_accuracies = []
lasso_cv_roc_aucs = []

for fold, (train_idx, val_idx) in enumerate(splits, 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    lasso_model = LogisticRegressionCV(
        Cs=10,
        cv=5,
        penalty='l1',
        solver='saga',
        max_iter=4000,
        random_state=630
    )
    lasso_model.fit(X_train_scaled, y_train)
    y_val_pred = lasso_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_val_pred)
    y_val_proba = lasso_model.predict_proba(X_val_scaled)
    roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr')
    lasso_cv_accuracies.append(acc)
    lasso_cv_roc_aucs.append(roc_auc)
    print(f"Lasso CV Fold {fold} accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")

print(f"Mean Lasso cross-validated accuracy (5 folds): {np.mean(lasso_cv_accuracies):.4f}")
print(f"Mean Lasso cross-validated ROC-AUC (5 folds): {np.mean(lasso_cv_roc_aucs):.4f}")