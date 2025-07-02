'''
extract_patient_id_for_split.py
Extracts all of the patient IDs from testData.csv and trainData.csv
and writes them as .csv files to be replicated in other models.
'''

import pandas as pd
from pathlib import Path

# Import data
path = Path.cwd().parent.parent
test = pd.read_csv(path / 'data' / 'processed' / 'testData.csv')
train = pd.read_csv(path / 'data' / 'processed' / 'trainData.csv')

# Select only the Patient IDs
testId = test['Patient.ID']
trainId = train['Patient.ID']

# Export to CSVs in the appropriate directory
testId.to_csv(path / 'data' / 'processed' / 'testDataPatientID.csv')
trainId.to_csv(path / 'data' / 'processed' / 'trainDataPatientID.csv')