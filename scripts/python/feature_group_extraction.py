'''
feature_group_extraction.py
'''

import pandas as pd
from pathlib import Path

# get the directory where this script is located
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# each of the groups we would like to categorize features by
FEATURE_GROUPS = ["Combining Tumor and FGT Enhancement",
                  "Tumor Size and Morphology",
                  "Tumor Enhancement Texture",
                  "Tumor Enhancement Spatial Heterogeneity",
                  "Tumor Enhancement Variation",
                  "Breast and FGT Volume Features",
                  "FGT Enhancement",
                  "Tumor Enhancement",
                  "FGT Enhancement Texture",
                  "FGT Enhancement Variation"]

# read in all relevant csv files
clinData = pd.read_csv(BASE_DIR / 'data/raw/clinicalData_clean.csv')
imagData = pd.read_csv(BASE_DIR /'data/raw/imagingFeatures.csv')
featureGroups = pd.read_csv(BASE_DIR / 'data/interim/imFeatures_and_feature_citations.csv',
                            skiprows=1,
                            names=['var', 'group'])

# join the clinical data and the imaging data
data = pd.merge(clinData, 
                imagData, 
                on = 'Patient ID',
                how = 'inner')

# need to have R-style column names
# where spaces and special characters are replaced with periods
data.columns = (
    data.columns
        .str.replace(r"[ ()=,]", ".", regex=True)
)

# display information about the dataframes
'''
print(data.head())
print(len(data.columns))
print(featureGroups.columns)
print(featureGroups.head())
'''

# transform the column for variable name
# use regex to remove [#] and ""
featureGroups['var'] = featureGroups['var'].str.replace(r'^\[\d+\]\s*"(.*)"$', r'\1', regex=True)

for group in FEATURE_GROUPS:
    # get the dataframe of columns pertinent to this group
    columns_df = featureGroups.query('group == @group')

    # flatten the dataframe to a list
    columns = columns_df['var'].to_list()

    # we also want to include Patient ID in each dataframe- can always ignore it
    columns.append('Patient.ID')
    
    tmp = data.filter(columns)

    # appropriately name the file and save to csv
    filename = group.replace(' ', '_') + '.csv'
    tmp.to_csv(BASE_DIR / f'data/raw/{filename}')