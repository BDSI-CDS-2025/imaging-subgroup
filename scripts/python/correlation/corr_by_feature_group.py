'''
corr_by_feature_group.py
For each feature category, look at all of the variables and create a correlation matrix.
If the correlation is higher then we can drop that feature.
Try to drop the highly correlated variables
Remove the variables that are potentially measuring similar things.

age, race, ethnicity, date of birth, menopause at diagnosis, tumor stage
'''

from pathlib import Path
import pandas as pd
import itertools
import numpy as np

# Paths
path = Path().cwd()
IMG = path / "data" / "raw" / "imagingFeatures.csv"
MAP = path / "data" / "interim" / "imFeatures_and_feature_citations.csv"

# Load data
img = pd.read_csv(IMG).dropna().drop(['Unnamed: 0', 'Patient ID'], axis = 1)
mapping = pd.read_csv(MAP, header=None, names=["var", "group"])

# Parameters
corr_threshold = 0.9

# Clean mapping variable names to match img columns
mapping["var"] = mapping["var"].str.replace(r'^\[\d+\]\s*"', '', regex=True).str.replace(r'"$', '', regex=True)
mapping["var"] = mapping["var"].str.replace('""', '"')  # Remove double quotes if present

# Build group-to-features dictionary
group_dict = mapping.groupby("group")["var"].apply(list).to_dict()

# Store correlated pairs and features to drop
correlated_pairs = []
features_to_drop = set()

for group, features in group_dict.items():

    # Subset the columns to only the ones that belong to the group
    # Only keep features present in img
    group_features = [f for f in features if f in img.columns]
    if len(group_features) < 2:
        continue

    sub_df = img[group_features]
    corr = sub_df.corr().abs()

    # Get upper triangle indices (excluding diagonal)
    upper = corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
    for i, j in itertools.combinations(range(len(group_features)), 2):
        f1, f2 = group_features[i], group_features[j]
        value = corr.iloc[i, j]
        if pd.notnull(value) and value >= corr_threshold:
            correlated_pairs.append({
                "feature_1": f1,
                "feature_2": f2,
                "correlation": value,
                "group": group
            })

            # Mark one feature for removal (arbitrary: drop f2)
            features_to_drop.add(f2)

# Output 1: CSV of correlated pairs
corr_pairs_df = pd.DataFrame(correlated_pairs)
print(f'corr_pairs_df: {corr_pairs_df.shape}')
corr_pairs_df.to_csv(path / "data" / "interim" / "highly_correlated_pairs.csv", index=False)

# Output 2: New imagingFeatures.csv with uncorrelated features
uncorrelated_img = img.drop(columns=list(features_to_drop), errors='ignore')
print(f'uncorrelated_img: {uncorrelated_img.shape}')
uncorrelated_img.to_csv(path / "data" / "interim" / "imagingFeatures_uncorrelated.csv", index=False)
