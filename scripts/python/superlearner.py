import pandas as pd
import numpy as np
import re

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Load training and test data ---
train_df = pd.read_csv("trainData.csv")
test_df = pd.read_csv("testData.csv")

# --- Fill missing numeric values ---
train_df = train_df.fillna(train_df.mean(numeric_only=True))
test_df = test_df.fillna(test_df.mean(numeric_only=True))

# --- Detect and drop messy columns ---
def is_messy_string(s):
    if isinstance(s, str):
        if re.search(r'\d+\s*[xX×]\s*\d+', s): return True
        if len(s.split()) > 5: return True
    return False

def detect_messy_columns(df, max_unique_ratio=0.9):
    messy_cols = []
    for col in df.select_dtypes(include='object').columns:
        unique_vals = df[col].dropna().unique()
        if any(is_messy_string(val) for val in unique_vals[:10]):
            messy_cols.append(col)
        elif len(unique_vals) / len(df) > max_unique_ratio:
            messy_cols.append(col)
    return messy_cols

messy_cols = list(set(detect_messy_columns(train_df) + detect_messy_columns(test_df)))
print(f"🧹 Dropping messy columns: {messy_cols}")
train_df = train_df.drop(columns=messy_cols, errors='ignore')
test_df = test_df.drop(columns=messy_cols, errors='ignore')

# --- Combine for consistent dummy encoding ---
train_df['__is_train'] = 1
test_df['__is_train'] = 0
combined = pd.concat([train_df, test_df], ignore_index=True)

# --- One-hot encode all object columns ---
combined = pd.get_dummies(combined, drop_first=True)

# --- Split back ---
train_df = combined[combined['__is_train'] == 1].drop(columns='__is_train')
test_df = combined[combined['__is_train'] == 0].drop(columns='__is_train')

# --- Separate features and target ---
target_col = 'Mol Subtype'
id_col = 'Patient ID'

X_train = train_df.drop(columns=[id_col, target_col], errors='ignore').values
y_train = train_df[target_col].astype(int).values

X_test = test_df.drop(columns=[id_col, target_col], errors='ignore').values
y_test = test_df[target_col].astype(int).values

# --- Standardize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Base + Meta Learners ---



# Base learners combo A
#base_learners = [
    #('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    #('svc', SVC(probability=True, kernel='rbf', C=1.0)),
#]


#Base learners combo B
#base_learners = [
    #('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    #('svc', SVC(probability=True, kernel='rbf', C=1.0)),
    #('nb', GaussianNB()),
    #('knn', KNeighborsClassifier(n_neighbors=5))
#]


#Base learners combo C

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('dt', DecisionTreeClassifier(max_depth=6)),
    ('gb', GradientBoostingClassifier(n_estimators=100)),
    ('ridge', RidgeClassifier())
]


#Base learners combo D
#base_learners = [
    #('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    #('svc', SVC(probability=True, kernel='poly', degree=2, C=1.0)),
    #('nb', GaussianNB()),
    #('knn', KNeighborsClassifier(n_neighbors=7)),
    #('ridge', RidgeClassifier()),
    #('dt', DecisionTreeClassifier(max_depth=5))
#]



#Meta learner A
#meta_learner = LogisticRegression(solver='lbfgs', max_iter=1000)


#Meta learner B
meta_learner = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)

ensemble = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    passthrough=False,
    n_jobs=-1
)

# --- Train/Evaluate ---
ensemble.fit(X_train_scaled, y_train)
y_pred = ensemble.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n StackingClassifier Accuracy for Mol Subtype prediction: {accuracy:.4f}")



# Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.show() 