import pandas as pd

# Load data (adjust paths as necessary)
clinData = pd.read_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/raw/clinicalData_clean.csv') 
imFeatures = pd.read_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/raw/imagingFeatures.csv')
# Remove 1st row of each DataFrame (as it contains extra information)
clinData = clinData.iloc[1:]
imFeatures = imFeatures.iloc[1:]
# Dimensions of clinical and imaging data
print("Clinical Data dimensions:", clinData.shape)
print("Imaging Features dimensions:", imFeatures.shape)

# Drop rows with missing values in imaging data
imFeatures = imFeatures.dropna()

# Merge clinical and imaging data on 'Patient ID' retaining all columns from clinical data
fullData = pd.merge(imFeatures, clinData, on='Patient ID', how='inner')

# Check dimensions of fullData
print("Full Data dimensions:", fullData.shape)

# Split fullData into training and testing sets
from sklearn.model_selection import train_test_split

trainData, testData = train_test_split(fullData, test_size=0.2, random_state=71)

# Check dimensions of training and testing sets
print("Training Data dimensions:", trainData.shape)
print("Testing Data dimensions:", testData.shape)
print(trainData.columns)

# Check for any overlap in Patient IDs between training and testing sets
train_patient_ids = trainData['Patient ID'].unique()
test_patient_ids = testData['Patient ID'].unique()
overlap = set(train_patient_ids).intersection(set(test_patient_ids))
if overlap:
    print("Overlap in Patient IDs between training and testing sets:", overlap)
else:
    print("No overlap in Patient IDs between training and testing sets.")

# Remove the column 'Unnamed: 0_x' if it exists in both DataFrames
if 'Unnamed: 0_x' in trainData.columns:
    trainData = trainData.drop(columns=['Unnamed: 0_x'])
if 'Unnamed: 0_x' in testData.columns:
    testData = testData.drop(columns=['Unnamed: 0_x'])
if 'Unnamed: 0_y' in trainData.columns:
    trainData = trainData.drop(columns=['Unnamed: 0_y'])
if 'Unnamed: 0_y' in testData.columns:
    testData = testData.drop(columns=['Unnamed: 0_y'])

# Check dimensions of training and testing sets after dropping the column
print("Training Data dimensions after dropping 'Unnamed: 0_x':", trainData.shape)
print("Testing Data dimensions after dropping 'Unnamed: 0_x':", testData.shape)


# where spaces and special characters are replaced with periods
trainData.columns = (
    trainData.columns
        .str.replace(r"[ ()=,]", ".", regex=True)
)
testData.columns = (
    testData.columns
        .str.replace(r"[ ()=,]", ".", regex=True)
)

# Save the training and testing sets to CSV files
trainData.to_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/processed/trainData.csv', index=False)
testData.to_csv('/Users/albertkang/Documents/BDSI_2025/imaging-subgroup/data/processed/testData.csv', index=False)