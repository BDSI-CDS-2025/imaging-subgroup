'''
per_feature_analysis_base_script.py
Base script to use for any per-feature analysis in Python.
'''

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

groups = ["Combining_Tumor_and_FGT_Enhancement.csv",
          "Tumor_Size_and_Morphology.csv",
          "Tumor_Enhancement_Texture.csv",
          "Tumor_Enhancement_Spatial_Heterogeneity.csv",
          "Tumor_Enhancement_Variation.csv",
          "Breast_and_FGT_Volume_Features.csv",
          "FGT_Enhancement.csv",
          "Tumor_Enhancement.csv",
          "FGT_Enhancement_Texture.csv",
          "FGT_Enhancement_Variation.csv"]

for group in groups:
    data = pd.read_csv(BASE_DIR / f'data/raw/{group}')

    # perform analysis here
    