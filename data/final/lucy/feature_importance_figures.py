'''
feature_importance_figures.py
Creates visualizations based on the data saved in
data/final/lucy/importance/best_models/{TARGET}_best_model_feature_importance.csv
'''

import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = Path.cwd() / "data" / "final" / "lucy" / "importance" / "best_models"
TARGETS = ['ER', 'PR', 'HER2']

for target in TARGETS:
    file = DATA_DIR / f'{target}_best_model_feature_importance.csv'
    df = pd.read_csv(file)
    df = df[:10]

    plt.figure(figsize=(10, 10))
    ax = sns.barplot(data=df,
                     x="feature",
                     y="importance_mean",
                     hue="importance_mean",
                     palette="Set1",
                     legend=False)
    plt.title(f'{target}: Feature Importance')
    plt.ylabel("Average Change in AUC when Permuted")
    plt.xlabel("Feature")
    plt.xticks(rotation=90)

    plt.tight_layout()
    save = DATA_DIR / f'{target}_feature_importance.png'
    plt.savefig(save, dpi=300)
    plt.close()
    print(f'Saved figure: {save}')