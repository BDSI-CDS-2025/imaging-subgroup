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
    file = DATA_DIR / f'{target}_best_model_feature_importance_uncorr.csv'
    df = pd.read_csv(file)
    df_top = df[:10]
    dfs[target] = df_top
    top_features.update(df_top['feature'].tolist())

top_features = sorted(top_features)

'''
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
    save = DATA_DIR / f'{target}_feature_importance_uncorr.png'
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
legend_save = DATA_DIR / "feature_color_key_uncorr.png"
plt.savefig(legend_save, dpi=300)
plt.close()
print(f'Saved color key: {legend_save}')
'''

def format_feature_label(feature):
    label = feature.replace('_', ' ')
    return label[:15] + '...' if len(label) > 15 else label

# --- Top 5 features per target and mean importance bar chart ---
# 1. Collect top 5 features for each target
top5_features = set()
top5_dict = {}
for target in TARGETS:
    df = dfs[target][:5]
    feats = df['feature'].tolist()
    top5_dict[target] = feats
    top5_features.update(feats)

print(f"Number of unique variables in the top 5 for each target: {len(top5_features)}")

# 2. Prepare data for plotting
plot_data = []
for feature in sorted(top5_features):
    for target in TARGETS:
        df = dfs[target]
        imp = df[df['feature'] == feature]['importance_mean']
        if not imp.empty:
            plot_data.append({'feature': feature, 'target': target, 'importance_mean': imp.values[0]})

plot_df = pd.DataFrame(plot_data)

# Add formatted labels to plot_df
plot_df['feature_label'] = plot_df['feature'].apply(format_feature_label)

# Calculate average importance for each feature
feature_means = plot_df.groupby('feature')['importance_mean'].mean()
ordered_features = feature_means.sort_values(ascending=False).index.tolist()
ordered_labels = [format_feature_label(f) for f in ordered_features]

# Plot with ordered features (horizontal bars) using formatted labels
plt.figure(figsize=(10, 12))
sns.barplot(
    data=plot_df,
    y='feature_label',
    x='importance_mean',
    hue='target',
    palette='Set2',
    order=ordered_labels
)

# Increase legend font size
plt.legend(fontsize=18) 

plt.title('Mean Importance of Top\n5 Features per Target\nOrdered by Average\n Importance', fontsize=30)
plt.xlabel('Mean Importance', fontsize=24, fontdict={'weight': 'bold'})
plt.ylabel('Feature', fontsize=24, fontdict={'weight': 'bold'})
plt.yticks(fontsize=24)
plt.xticks(fontsize=15)

plt.tight_layout(pad=5)
plt.subplots_adjust(right=.85)  # Add more space on the right

save_path = DATA_DIR / 'top5_features_mean_importance_ordered_uncorr_horizontal.png'
plt.savefig(save_path, dpi=300)
plt.close()
print(f"Saved ordered top 5 features mean importance horizontal bar")
