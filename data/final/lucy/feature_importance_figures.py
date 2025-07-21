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

# 1. Collect top features
top_features = set()
dfs = {}
for target in TARGETS:
    file = DATA_DIR / f'{target}_best_model_feature_importance.csv'
    df = pd.read_csv(file)
    df_top = df[:10]
    dfs[target] = df_top
    top_features.update(df_top['feature'].tolist())

top_features = sorted(top_features)

# 2. Assign colors
palette = sns.color_palette("hls", n_colors=len(top_features))  # or "tab20"
feature_to_color = dict(zip(top_features, palette))

# 3. Plot with consistent colors
for target in TARGETS:
    df = dfs[target]
    plt.figure(figsize=(10, 10))
    ax = sns.barplot(
        data=df,
        x="feature",
        y="importance_mean",
        hue="feature",
        palette=feature_to_color,
        dodge=False,
        legend=False,
    )
    plt.title(f'{target}: Feature Importance')
    plt.ylabel("Average Change in AUC when Permuted")
    plt.xlabel("Feature")
    plt.xticks(rotation=90)
    plt.tight_layout()
    save = DATA_DIR / f'{target}_feature_importance.png'
    plt.savefig(save, dpi=300)
    plt.close()
    print(f'Saved figure: {save}')

# 4. Output legend/key as a figure
plt.figure(figsize=(8, len(top_features) * 0.4))
for i, (feature, color) in enumerate(feature_to_color.items()):
    plt.barh(i, 1, color=color)
    plt.text(0.05, i, feature, va='center', ha='left', fontsize=12)
plt.yticks([])
plt.xticks([])
plt.title("Feature Color Key")
plt.tight_layout()
legend_save = DATA_DIR / "feature_color_key.png"
plt.savefig(legend_save, dpi=300)
plt.close()
print(f'Saved color key: {legend_save}')