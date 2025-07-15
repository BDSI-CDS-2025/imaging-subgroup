'''
config.py
Contain information about file locations so that models are able to startup
with hyperparameter configuraiton.
'''

from pathlib import Path

path = Path.cwd()

# Temporary with only one dataset for debugging.
SETUP2 = {
    'PC1': {
        'X': path / "data" / "interim" / "pc_by_feature_group_for_patients.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_pc1.json"
    }
}

# Delete SETUP and change name when ready finally.
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
        'X' : path / "data" / "interim" / "pc_90percent.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_ninety.json"
    },
    'ALL_IMG': {
        'X': path / "data" / "raw" / "imagingFeatures.csv",
        'hyperparameters_file': path / "notebooks" / "modeling" / "optimal_params_all_image_features.json"
    },
    'VARS_IN_PC1' : {
        'X': path / "data" / "interim" / "patient_top_loading_factor_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "optimal_params_vars_in_pc1.json"
    },
    'VARS_IN_PC_1_3' : {
        'X': path / "data" / "interim" / "patient_top_three_loading_factors_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "optimal_params_vars_in_pc1_3.json"
    },
    'VARS_IN_PC_90' : {
        'X': path / "data" / "interim" / "patient_ninety_percent_factors_by_subgroup.csv",
        'hyperparameters_file' : path / "notebooks" / "modeling" / "optimal_params_vars_in_pc_90.json"
    }

}