'''
config_per_var_fam.py
Contain information about file locations so that models are able to startup
with hyperparameter configuraiton.
'''

# config_per_var_fam.py
from pathlib import Path
data_root = Path(__file__).parent.parent.parent / "data" / "raw"
modeling_root = Path(__file__).parent.parent.parent / "notebooks" / "modeling"

# Delete SETUP and change name when ready finally.
SET = {
    'Breast_and_FGT_Volume_Features': {
        'X': data_root / "Breast_and_FGT_Volume_Features.csv",
        'hyperparameters_file': modeling_root / "optimal_params_Breast_and_FGT_Volume_Features.json"
    },
    'Combining_Tumor_and_FGT_Enhancement' : {
        'X': data_root / "Combining_Tumor_and_FGT_Enhancement.csv",
        'hyperparameters_file': modeling_root / "optimal_params_Combining_Tumor_and_FGT_Enhancement.json"
    },
    'FGT_Enhancement' : {
        'X' : data_root / "FGT_Enhancement.csv",
        'hyperparameters_file': modeling_root / "optimal_params_FGT_Enhancement.json"
    },
    'FGT_Enhancement_Texture': {
        'X': data_root / "FGT_Enhancement_Texture.csv",
        'hyperparameters_file': modeling_root / "optimal_params_FGT_Enhancement_Texture.json"
    },
    'FGT_Enhancement_Variation' : {
        'X': data_root / "FGT_Enhancement_Variation.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_FGT_Enhancement_Variation.json"
    },
    'Tumor_Enhancement' : {
        'X': data_root / "Tumor_Enhancement.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_Tumor_Enhancement.json"
    },
    'Tumor_Enhancement_Spatial_Heterogeneity' : {
        'X': data_root / "Tumor_Enhancement_Spatial_Heterogeneity.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_Tumor_Enhancement_Spatial_Heterogeneity.json"
    },
    'Tumor_Enhancement_Texture' : {
        'X': data_root / "Tumor_Enhancement_Texture.csv",
        'hyperparameters_file' : modeling_root / "optimal_params_Tumor_Enhancement_Texture.json"
    },
    'Tumor_Enhancement_Variation': {
        'X': data_root / "Tumor_Enhancement_Variation.csv",
        'hyperparameters_file': modeling_root / "optimal_params_Tumor_Enhancement_Variation.json"
    },
    'Tumor_Size_and_Morphology': {
        'X': data_root / "Tumor_Size_and_Morphology.csv",
        'hyperparameters_file': modeling_root / "optimal_params_Tumor_Size_and_Morphology.json"
    }
}