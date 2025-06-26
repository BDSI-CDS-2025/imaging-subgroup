'''
random_forest_tumor.py
Creating a random forest classifier and then using the SHAP
module to try and explain the model outputs.

Getting started with Shap:
https://medium.com/biased-algorithms/shap-values-for-random-forest-1150577563c9
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import shap
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# always must initialize the shap library
shap.initjs()

# load only tumor data
tumor_enhancement_spatial = pd.read_csv(BASE_DIR / 'data/raw/Tumor_Enhancement_Spatial_Heterogeneity.csv')
tumor_enhancement_texture = pd.read_csv(BASE_DIR / 'data/raw/Tumor_Enhancement_Texture.csv')

# merge into one dataframe, removing the patient ID column
tumor_data = (
    pd.merge(tumor_enhancement_spatial, 
            tumor_enhancement_texture,
            how='inner',
            on='Patient.ID')
    .drop('Patient.ID', axis=1)
)

# select feature and target variables
X = tumor_data
y = pd.read_csv(BASE_DIR / 'data/raw/clinicalData_clean.csv')['ER']

X_train, X_test, y_train, y_test = train_test_split(X, y)

# build the random forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# get the accuracy on the test data
accuracy = clf.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

# explain using shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer(X_test)

# should be of the form (num_samples, num_features)
# is of the form (num_samples, num_features, num_outputs)
print(shap_values.shape)

# for the first observation
pred_class = clf.predict(X_test)
shap.plots.waterfall(shap_values[0, :, pred_class[0]],
                     max_display=10)
