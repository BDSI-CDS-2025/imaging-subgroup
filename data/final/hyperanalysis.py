'''
hyperanalysis.py
A script used for metaanalysis of the engineers.
'''

import pandas as pd
import glob

'''
Find the paitient IDs of the patients with the lowest
and highest values for PC3_FGT_Enhancement_Texture.
df = pd.read_csv("PC_90_with_clin.csv")
print(min(df['PC3_FGT_Enhancement_Texture']))
print(max(df['PC3_FGT_Enhancement_Texture']))
'''

# MIN VALUE: Breast_MRI_824
# MAX VALUE: Breast_MRI_607

csv_files = glob.glob("*.csv")
for file in csv_files:
    current = pd.read_csv(file)
    print(f'{file}: {len(current)}')