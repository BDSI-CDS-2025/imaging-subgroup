import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Load data
clinData = pd.read_csv('data/raw/clinicalData_clean.csv')
imFeatures = pd.read_csv('data/raw/imagingFeatures.csv')
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

# 5-fold cross-validated multinomial logistic regression, with 1 fold held out as test set
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=630)
splits = list(skf.split(X, y))

# Use the last fold as the test set, the first 4 as cross-validation
cv_splits = splits[:4]
test_split = splits[4]

cv_accuracies = []

for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)  # removed multi_class
    logit_model.fit(X_train_scaled, y_train)
    y_pred = logit_model.predict(X_val_scaled)
    acc = accuracy_score(y_val, y_pred)
    cv_accuracies.append(acc)
    print(f"CV Fold {fold} accuracy: {acc:.4f}")

print(f"Mean cross-validated accuracy (4 folds): {np.mean(cv_accuracies):.4f}")

# Evaluate on held-out test set
train_idx, test_idx = test_split
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
logit_model = LogisticRegression(solver='lbfgs', max_iter=1000)  # removed multi_class
logit_model.fit(X_train_scaled, y_train)
y_pred = logit_model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, y_pred)
print(f"Held-out test set accuracy: {test_acc:.4f}")