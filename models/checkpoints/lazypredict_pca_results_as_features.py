'''
lazypredict_er.py
Compare different model's ability to predict ER based on the PC1 coordinates.

https://www.kaggle.com/code/suneelpatel/compare-ml-models-with-few-lines-of-code
'''
from pathlib import Path
import pandas as pd
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# set up the data, choose the column we would like to predict
data = pd.read_csv(BASE_DIR / 'data/interim/pc_by_feature_group_for_patients.csv')
all = pd.read_csv(BASE_DIR / 'data/raw/clinicalData_clean.csv')
X = data.drop('Patient.ID', axis=1)
y = all['HER2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# create the regression models
reg = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# print out the model performance, sorted by accuracy
print(models.sort_values(by='Accuracy'))