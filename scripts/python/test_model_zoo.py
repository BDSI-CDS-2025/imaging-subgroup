'''
test_model_zoo.py
For each of the hyperparameter configurations as described in
/notebooks/exploration/modeling, reports the test data.

To run:
python3 scripts/python/test_model_zoo.py
'''

import pandas as pd
import numpy as np
import json
import seaborn as sns
import os

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import warnings

path = Path.cwd()

SETUP = {
    'PC1': {
        'X': path / "data" / "interim" / "pc_by_feature_group_for_patients.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_pc1.json"
    },
    'PC1_3': {
        'X': path / "data" / "interim" / "pc1_to_3_by_feature_group_for_patients.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_pc1_to_3.json"
    },
    'PC_90' : {
        'X' : path / "data" / "interim" / "pc_90percent.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_ninety.json"
    },
    'ALL_IMG': {
        'X': path / "data" / "raw" / "imagingFeatures.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_all_image_features.json"
    },
    'VARS_IN_PC1' : {
        'X': path / "data" / "interim" / "patient_top_loading_factor_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_vars_in_pc1.json"
    },
    'VARS_IN_PC_1_3' : {
        'X': path / "data" / "interim" / "patient_top_three_loading_factors_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_vars_in_pc1_3.json"
    },
    'VARS_IN_PC_90' : {
        'X': path / "data" / "interim" / "patient_ninety_percent_factors_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "mol-subtype" / "optimal_params_vars_in_pc_90.json"
    }

}

warnings.filterwarnings("ignore") # don't show any warnings

TARGET = 'Mol Subtype'
path = Path.cwd()
RESULT_DIR = path / "scripts" / "python" / "all_model_results" / "MOL-SUBTYPE"

LABEL_TO_SUBTYPE = {
    '0': 'luminal-like',
    '1': 'ER/PR pos, HER2 pos',
    '2': 'HER2 pos',
    '3': 'trip neg'
}

def load_data(X_path, train_ids_path, test_ids_path, clin_path):
    trainPatientID = pd.read_csv(train_ids_path).rename(columns={'Patient.ID': 'Patient ID'}, errors='ignore')
    testPatientID = pd.read_csv(test_ids_path).rename(columns={'Patient.ID': 'Patient ID'}, errors='ignore')
    features = pd.read_csv(X_path).rename(columns={'Patient.ID': 'Patient ID'}, errors='ignore')
    clin = pd.read_csv(clin_path)
    data = features.merge(clin[['Patient ID', TARGET]], on='Patient ID', how='inner')
    y = data[[TARGET, 'Patient ID']]

    # Always fit a LabelEncoder
    le = LabelEncoder()
    y[TARGET] = le.fit_transform(y[TARGET].astype(str))  # Cast to str to treat all as categories

    yTrain = y[y['Patient ID'].isin(trainPatientID['Patient ID'])][TARGET]
    yTest = y[y['Patient ID'].isin(testPatientID['Patient ID'])][TARGET]
    XTrain = data[data['Patient ID'].isin(trainPatientID['Patient ID'])].drop(['Patient ID', TARGET], axis=1)
    XTest = data[data['Patient ID'].isin(testPatientID['Patient ID'])].drop(['Patient ID', TARGET], axis=1)
    return XTrain, XTest, yTrain, yTest, le if 'le' in locals() else None

def plot_and_save_roc(y_true, y_score, n_classes, le, model_name, dataset_name, save_dir):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = roc_auc_score(y_true == i, y_score[:, i])
        class_label = le.inverse_transform([i])[0]
        plt.plot(fpr[i], tpr[i], label=f'Class {LABEL_TO_SUBTYPE[class_label]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name} on {dataset_name}')
    plt.legend()
    plt.tight_layout()
    out_path = save_dir / f"{model_name}_roc.png"
    plt.savefig(out_path)
    plt.close()
    return roc_auc

def plot_and_save_confusion_matrix(cm, le, model_name, dataset_name, save_dir):
    plt.figure(figsize=(7, 6))
    labels = le.inverse_transform(range(cm.shape[0]))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.tight_layout()
    out_path = save_dir / f"{model_name}_confusion.png"
    plt.savefig(out_path)
    plt.close()

def test_all():
    results = {}

    # Iterate through each dataset in SETUP
    for dataset_name, setup in SETUP.items():
        print(f"\nTesting on dataset: {dataset_name}")
        XTrain, XTest, yTrain, yTest, le = load_data(
            setup['X'],
            path / "data" / "processed" / "trainDataPatientID.csv",
            path / "data" / "processed" / "testDataPatientID.csv",
            path / "data" / "raw" / "clinicalData_clean.csv"
        )

        # Load optimal hyperparameters
        with open(setup['hyperparameters_file'], 'r') as file:
            model_info = json.load(file)
        
        dataset_results = {}

        # Create output directory for this dataset
        dataset_dir = RESULT_DIR / dataset_name
        os.makedirs(dataset_dir, exist_ok=True)

        # Fit each model on train
        for model in model_info:
            model_name = model['model']
            params = model['best_params']
            
            if model_name == 'RandomForest':
                clf = RandomForestClassifier(**params)
            elif model_name == 'XGBoost':
                clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
            elif model_name == 'MLP':
                clf = MLPClassifier(max_iter=2000, **params)
            else:
                continue
            clf.fit(XTrain, yTrain)
            
            # Evaluate on test
            y_pred = clf.predict(XTest)
            y_proba = clf.predict_proba(XTest)
            acc = accuracy_score(yTest, y_pred)
            auc = roc_auc_score(yTest, y_proba, multi_class='ovr')
            cm = confusion_matrix(yTest, y_pred)
            print(f"{model_name} - Accuracy: {acc:.3f}, AUC: {auc:.3f}")
            
            # Plot ROC
            plot_and_save_roc(yTest, y_proba, len(le.classes_), le, model_name, dataset_name, dataset_dir)
            # Save confusion matrix heatmap
            plot_and_save_confusion_matrix(cm, le, model_name, dataset_name, dataset_dir)

            dataset_results[model_name] = {
                'accuracy': acc,
                'auc': auc,
                'confusion_matrix': cm.tolist()
            }
        results[dataset_name] = dataset_results
    
    # Optionally save results
    with open(RESULT_DIR / 'all_model_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved all results to all_model_test_results.json")

if __name__ == "__main__":
    test_all()