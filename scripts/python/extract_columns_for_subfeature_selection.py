'''
extract_columns_for_subfeature_selection.py

For each of the given files, selects only the
relevant columns and creates a new .csv file with the patient
features only for those selected columns.

To run:
python3 scripts/python/extract_columns_for_subfeature_selection.py
'''

from pathlib import Path
import pandas as pd

SOURCE_DIR = Path(__file__).parent.parent.parent / "results" / "reports"
IMG = Path(__file__).parent.parent.parent / "data" / "raw" / "imagingFeatures.csv"
DEST_DIR = Path(__file__).parent.parent.parent / "data" / "interim"
FEATURE_FILES = ["ninety_percent_factors_by_subgroup.csv",
                 "top_loading_factor_by_subgroup.csv",
                 "top_three_loading_factors_by_subgroup.csv"]

img = pd.read_csv(IMG)
img.columns = (
    img.columns
        .str.replace(r"[ ()=,]", ".", regex=True)
)

for f in FEATURE_FILES:
    df = pd.read_csv(SOURCE_DIR / f)

    # Select only the columns of the image dataframe that
    # are in the 'variable' column of df
    keep = df['variable'].tolist()
    keep.append('Patient.ID')
    selected_img = img[keep]

    out_file = "patient_" + f
    selected_img.to_csv(DEST_DIR / out_file, index=False)
    