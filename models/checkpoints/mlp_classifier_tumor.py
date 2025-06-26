'''
mlp_classifier_tumor.py
Builds a multilayer perceptron to predict ER based on
the feature groups corresponding to tumor properties,
and then interpret the model using the SHAP library.
'''

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pandas as pd
import shap

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# always must initialize the shap library
shap.initjs()

# to use all data
'''
all_data = (
    pd.read_csv(BASE_DIR /'data/raw/joined_data.csv')
    .drop(['Unnamed: 0', 'Unnamed:.0_x', 'Patient.ID'], axis=1)
)
X = all_data.drop('HER2', axis=1)
y = all_data['HER2']
'''

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

# scale the data so that it works with mlp
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# multilayer perceptron
mlp = MLPClassifier(solver='lbfgs', 
                    alpha=1, 
                    hidden_layer_sizes=(5, 2),
                    random_state = 1)

# fit the model and output accuracy
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
print(f'accuracy: {accuracy}')