'''
extract_test_result_table.py
Reads the results from all_model_test_results.json and displays
a nice table.

To run:
python3 scripts/python/extract_test_result_table.py
'''

import json
import pandas as pd
from pathlib import Path

path = Path.cwd()
DEST = path / "scripts" / "python" / "all_model_results"
SOURCE_FILE = DEST / "all_model_test_results.json"

# Load the results JSON
with open(SOURCE_FILE, "r") as f:
    results = json.load(f)

# Prepare a list to collect rows
rows = []

# For each dataset, collect accuracy and AUC for each model
for dataset, models in results.items():
    row = {"Dataset": dataset}
    for model in ["RandomForest", "XGBoost", "MLP"]:
        if model in models:
            row[f"{model}_Accuracy"] = models[model]["accuracy"]
            row[f"{model}_AUC"] = models[model]["auc"]
        else:
            row[f"{model}_Accuracy"] = None
            row[f"{model}_AUC"] = None
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)
df = df.set_index("Dataset")
print(df)

# Optionally, save to CSV
df.to_csv("../all_model_test_results_table.csv")