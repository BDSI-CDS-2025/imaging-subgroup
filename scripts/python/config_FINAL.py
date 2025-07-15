'''
config_final.py
Contain information about file locations so that models are able to startup
with hyperparameter configuraiton.
'''

from pathlib import Path
data_root = Path(__file__).parent.parent.parent / "data" / "final"
modeling_root = Path(__file__).parent.parent.parent / "notebooks" / "modeling"


# Delete SETUP and change name when ready finally.
SETUP = {
    'ALL_IMG': {
        'X': data_root / "ALL_IMG.csv",
        'hyperparameters_file': modeling_root / "optimal_params_ALL_IMG.json"
    },
    'ALL_IMG_with_clin': {
        'X': data_root / "ALL_IMG_with_clin.csv",
        'hyperparameters_file': modeling_root / "optimal_params_ALL_IMG_with_clin.json"
    },
    'CL_UNCORR' : {
        'X' : data_root / "CL_UNCORR.csv",
        'hyperparameters_file': modeling_root / "optimal_params_CL_UNCORR.json"
    },
    'CL_UNCORR_with_clin': {
        'X': data_root / "CL_UNCORR_with_clin.csv",
        'hyperparameters_file': modeling_root / "optimal_params_CL_UNCORR_with_clin.json"
    },
    'PC1_3' : {
        'X': data_root / "PC1_3.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_vars_PC1_3.json"
    },
    'PC1_3_with_clin' : {
        'X': data_root / "PC1_3_with_clin.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_PC1_3_with_clin.json"
    },
    'PC1' : {
        'X': data_root / "PC1.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_PC1.json"
    },
    'PC1_with_clin' : {
        'X': data_root / "PC1_with_clin.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_PC1_with_clin.json"
    },
    'PC_90': {
        'X': data_root / "PC_90.csv",
        'hyperparameters_file': modeling_root / "optimal_params_PC_90.json"
    },
    'PC_90_with_clin': {
        'X': data_root / "PC_90_with_clin.csv",
        'hyperparameters_file': modeling_root / "optimal_params_PC_90_with_clin.json"
    },
    'VARS_IN_PC1' : {
        'X' : data_root / "VARS_IN_PC1.csv",
        'hyperparameters_file': modeling_root / "optimal_params_VARS_IN_PC1.json"
    },
    'VARS_IN_PC1_with_clin': {
        'X': data_root / "VARS_IN_PC1_with_clin.csv",
        'hyperparameters_file': modeling_root / "optimal_params_VARS_IN_PC1_with_clin.json"
    },
    'VARS_IN_PC1_3' : {
        'X': data_root / "VARS_IN_PC1_3.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_VARS_IN_PC1_3.json"
    },
    'VARS_IN_PC_1_3_with_clin' : {
        'X': data_root / "VARS_IN_PC_1_3_with_clin.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_VARS_IN_PC_1_3_with_clin.json"
    },
    'VARS_IN_PC_90' : {
        'X': data_root / "VARS_IN_PC_90.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_vars_in_pc_90.json"
    },
    'VARS_IN_PC_90_with_clin' : {
        'X': data_root / "VARS_IN_PC_90_with_clin.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_VARS_IN_PC_90_with_clin.json"
    }

}