'''
config.py
Contain information about file locations so that models are able to startup
with hyperparameter configuraiton.
'''

from pathlib import Path

path = Path.cwd()

SETUP = {
    'PC1': {
        'X': path / "data" / "interim" / "pc_by_feature_group_for_patients.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_pc1.json"
    },
    'PC1_3': {
        'X': path / "data" / "interim" / "pc1_to_3_by_feature_group_for_patients.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_pc1_to_3.json"
    },
    'PC_90' : {
        'X' : path / "data" / "interim" / "pc90percent.csv",
        'hyperparameters_file': path / "notebooks" / "modleing" / "optimal_params_ninety.json"
    },
    'ALL_IMG': {
        'X': path / "data" / "raw" / "imagingFeatures.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_all_image_features.json"
    }
}