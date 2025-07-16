'''
A script to find the paitient IDs of the patients with the lowest
and highest values for PC3_FGT_Enhancement_Texture.
'''

import pandas as pd
df = pd.read_csv("PC_90_with_clin.csv")
print(min(df['PC3_FGT_Enhancement_Texture']))
print(max(df['PC3_FGT_Enhancement_Texture']))

# MIN VALUE: Breast_MRI_824
# MAX VALUE: Breast_MRI_607