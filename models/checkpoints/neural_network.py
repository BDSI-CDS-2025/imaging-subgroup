from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
data = (
    pd.read_csv(BASE_DIR /'data/raw/joined_data.csv')
    .drop(['Unnamed: 0', 'Unnamed:.0_x'], axis=1)
)

X = data.drop('HER2', axis=1)
y = data['HER2']

X_train, X_test, y_train, y_test = train_test_split(X, y)

clf = MLPClassifier(solver='lbfgs', 
                    alpha=1, 
                    hidden_layer_sizes=(5, 2),
                    random_state = 1)