'''
merge_with_clinical.py
Produces a dataset merged with the four pre-biopsy clinical features
for each of the 8 datasets.

age, race, ethnicity, date of birth, tumor stage
To run: python3 scripts/python/data_cleaning/merge_with_clinical.py

Menopause
0 = pre
1 = post
2 = N/A

Date of Birth (days) takes date of diagnosis as day 0
'''

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SETUP

path = Path().cwd()

CLIN_PATH = path / "data" / "raw" / "clinicalData_clean.csv"
COLS_TO_MERGE = ["Patient ID", "Menopause (at diagnosis)", "Race and Ethnicity", "Date of Birth (Days)", "Staging(Tumor Size)# [T]"]
RES_PATH = path / "data" / "interim" / "with_clin"
from pathlib import Path

# Change so that can import from the appropriate path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import SETUP
clin = pd.read_csv(CLIN_PATH).drop(['Unnamed: 0'], axis = 1, errors = 'ignore')
clin.rename(columns = {'Patient.ID' : 'Patient ID'}, errors = 'ignore')
clin = clin[COLS_TO_MERGE]

# Encode 2 for menopause as NA
print(f'number menopause NA: {len(clin[clin["Menopause (at diagnosis)"] == 2])}')
clin[clin["Menopause (at diagnosis)"] == 2] = np.nan

# Change race to be only Black, white, other (where other includes NA)
clin.loc[(clin["Race and Ethnicity"] != 1) | (clin["Race and Ethnicity"] != 2), "Race and Ethnicity"] = 0

# Change age to be nonnegative
clin['Date of Birth (Days)'] = abs(clin['Date of Birth (Days)'])

# There are 900 rows left after dropping NA
clin = clin.dropna()

'''
Now:
Menolause (no NA)
0 = pre
1 = post
Age is the number of days from birth to diagnosis
Race and Ethnicity
0 = other
1 = white
2 = Black
'''

print(clin.head)

# For each of the datasets we would like to duplicate
for dataset_name in SETUP.keys():
    input_file = SETUP[dataset_name]['X']
    df = pd.read_csv(input_file)
    df = df.rename(columns={'Patient.ID' : 'Patient ID'}, errors = 'ignore')
    
    joined_with_clin = df.merge(clin, how = 'inner', on = 'Patient ID').dropna()
    joined_with_clin.drop(['Unnamed: 0'], axis = 1, errors = 'ignore')
    dest = RES_PATH / (dataset_name + '_with_clin.csv')
    joined_with_clin.to_csv(dest)
    print(f'length of merged_{dataset_name} is {len(joined_with_clin)}')
