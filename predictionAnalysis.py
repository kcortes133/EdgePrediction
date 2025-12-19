import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import glob
import os

# ----------------------
# Parameters
# ----------------------
predictions_glob = '*/HMI_top.csv'  # glob pattern for HMI files


# ----------------------

# Function to extract rank from classification string
def extract_rank(classification):
    match = re.search(r'rank=(\d+)', str(classification))
    return int(match.group(1)) if match else None


# Store data
all_predictions = []
all_gt_summary = []

# Find all HMI prediction files
prediction_files = glob.glob(predictions_glob)
if not prediction_files:
    raise ValueError(f"No prediction files found with pattern: {predictions_glob}")

for hmi_file in prediction_files:
    folder = os.path.dirname(hmi_file)
    approach_name = os.path.basename(folder)  # folder name as label

    # Prediction file
    pred_df = pd.read_csv(hmi_file)
    if pred_df.empty:
        print(f"Warning: {hmi_file} is empty, skipping.")
        continue

    # Extract rank and TopN
    pred_df['rank'] = pred_df['classification'].apply(extract_rank)
    pred_df['file'] = approach_name
    pred_df['TopN'] = pred_df['classification'].apply(
        lambda x: re.match(r'Top\d+', str(x)).group(0) if pd.notna(x) else 'Unknown')
    pred_df['Top10_flag'] = pred_df['TopN'] == 'Top10'

    all_predictions.append(pred_df)

    # Corresponding ground truth file
    gt_file = os.path.join(folder, 'top10results.csv')
    if os.path.exists(gt_file):
        gt_df = pd.read_csv(gt_file)
        tp_count = (gt_df['classification'] == 'TP').sum()
        fn_count = (gt_df['classification'] == 'FN').sum()
        total = len(gt_df)
        all_gt_summary.append({'Approach': approach_name, 'TP': tp_count, 'FN': fn_count, 'Total': total})
    else:
        print(f"Warning: Ground truth file not found in {folder}, skipping GT summary for this approach.")

# Combine all prediction data
combined_df = pd.concat(all_predictions, ignore_index=True)

# ----------------------
# Plotting
# ----------------------
sns.set(style='whitegrid', palette='muted', font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# 1. Histogram of ranks
sns.histplot(data=combined_df, x='rank', hue='file', bins=30, element='step', stat='count', common_norm=False,
             ax=axes[0])
axes[0].set_title('Histogram of Gene Prediction Ranks')
axes[0].set_xlabel('Rank of True Gene')
axes[0].set_ylabel('Count')

# 2. CDF of ranks
sns.ecdfplot(data=combined_df, x='rank', hue='file', ax=axes[1])
axes[1].set_title('CDF of Gene Prediction Ranks')
axes[1].set_xlabel('Rank of True Gene')
axes[1].set_ylabel('Cumulative Probability')

# 3. Score vs Rank scatter with Top10 highlighted per approach
palette = sns.color_palette("tab10", n_colors=combined_df['file'].nunique())
color_map = dict(zip(combined_df['file'].unique(), palette))

for file in combined_df['file'].unique():
    subset = combined_df[combined_df['file'] == file]

    # Normal points (non-Top10)
    axes[2].scatter(subset[~subset['Top10_flag']]['rank'],
                    subset[~subset['Top10_flag']]['score'],
                    alpha=0.6,
                    color=color_map[file],
                    label=file)

    # Top10 points: same color but larger with black edge
    axes[2].scatter(subset[subset['Top10_flag']]['rank'],
                    subset[subset['Top10_flag']]['score'],
                    s=50, edgecolor='black', facecolor=color_map[file], linewidth=1.5,
                    label=f'{file} Top10')

axes[2].set_title('Prediction Score vs True Gene Rank (Top10 Highlighted)')
axes[2].set_xlabel('Rank of True Gene')
axes[2].set_ylabel('Prediction Score')
axes[2].legend(loc='best')

# 4. Top-N counts
sns.countplot(data=combined_df, x='TopN', hue='file', palette=palette , ax=axes[3])
axes[3].set_title('Number of Predictions by Top-N Category')
axes[3].set_xlabel('Top-N Category')
axes[3].set_ylabel('Count')

plt.tight_layout()
plt.show()

# ----------------------
# Summary statistics
# ----------------------
summary = combined_df.groupby('file')['rank'].describe()[['count', 'mean', '50%', 'min', 'max']]
summary.rename(columns={'50%': 'median'}, inplace=True)
print("\nPrediction rank statistics by approach:")
print(summary)

topn_summary = combined_df.groupby(['file', 'TopN']).size().unstack(fill_value=0)
print("\nTop-N counts by prediction approach:")
print(topn_summary)

top10_summary = combined_df[combined_df['Top10_flag']].groupby('file').size()
print("\nNumber of Top10 predictions by approach:")
print(top10_summary)

# Ground truth summary
gt_summary_df = pd.DataFrame(all_gt_summary)
print("\nGround truth TP/FN summary per approach:")
print(gt_summary_df)



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import re

# ==============================
# USER PARAMETERS
# ==============================
BASE_GLOB = '*/HMI_top.csv'
NUM_CANDIDATES_PER_DISEASE = 100   # <-- your defined TN space
SAVE_FIGURES = False
OUT_PREFIX = 'prediction_summary'

# ==============================
# HELPERS
# ==============================
def extract_rank(classification):
    m = re.search(r'rank=(\d+)', str(classification))
    return int(m.group(1)) if m else None

def extract_topn(classification):
    m = re.match(r'(Top\d+)', str(classification))
    return m.group(1) if m else 'Unknown'

# ==============================
# LOAD DATA
# ==============================
prediction_files = glob.glob(BASE_GLOB)
if not prediction_files:
    raise ValueError(f'No prediction files found with pattern: {BASE_GLOB}')

all_preds = []
confusion_rows = []

for hmi_file in prediction_files:
    folder = os.path.dirname(hmi_file)
    approach = os.path.basename(folder)

    # ---- Predictions ----
    pred_df = pd.read_csv(hmi_file)
    if pred_df.empty:
        print(f'Skipping empty file: {hmi_file}')
        continue

    pred_df['rank'] = pred_df['classification'].apply(extract_rank)
    pred_df['TopN'] = pred_df['classification'].apply(extract_topn)
    pred_df['Top10_flag'] = pred_df['TopN'] == 'Top10'
    pred_df['Approach'] = approach
    all_preds.append(pred_df)

    # ---- Ground truth ----
    gt_file = os.path.join(folder, 'top10results.csv')
    if not os.path.exists(gt_file):
        print(f'WARNING: Missing GT file for {approach}')
        continue

    gt_df = pd.read_csv(gt_file)

    TP = (gt_df['classification'] == 'TP').sum()
    FN = (gt_df['classification'] == 'FN').sum()

    # FP inferred from prediction space
    FP = len(pred_df) - TP

    # TN from defined candidate space
    n_diseases = gt_df['object'].nunique()
    TOTAL = n_diseases * NUM_CANDIDATES_PER_DISEASE
    TN = TOTAL - TP - FN - FP

    confusion_rows.append({
        'Approach': approach,
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'TN': TN
    })

combined_df = pd.concat(all_preds, ignore_index=True)
confusion_df = pd.DataFrame(confusion_rows)

# ==============================
# METRICS
# ==============================
confusion_df['Precision'] = confusion_df['TP'] / (confusion_df['TP'] + confusion_df['FP'])
confusion_df['Recall'] = confusion_df['TP'] / (confusion_df['TP'] + confusion_df['FN'])
confusion_df['Specificity'] = confusion_df['TN'] / (confusion_df['TN'] + confusion_df['FP'])
confusion_df['Accuracy'] = (
    confusion_df['TP'] + confusion_df['TN']
) / (
    confusion_df[['TP','FP','FN','TN']].sum(axis=1)
)
confusion_df['F1'] = 2 * (
    confusion_df['Precision'] * confusion_df['Recall']
) / (
    confusion_df['Precision'] + confusion_df['Recall']
)

print("\n=== Confusion Matrix + Metrics ===")
print(confusion_df.round(4))

# ==============================
# PLOTTING
# ==============================
sns.set(style='whitegrid', font_scale=1.1)
palette = dict(zip(
    combined_df['Approach'].unique(),
    sns.color_palette('tab10', combined_df['Approach'].nunique())
))

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# ---- (1) Rank histogram ----
sns.histplot(
    data=combined_df,
    x='rank',
    hue='Approach',
    bins=30,
    element='step',
    common_norm=False,
    ax=axes[0]
)
axes[0].set_title('Rank Distribution')
axes[0].set_xlabel('Rank of True Gene')

# ---- (2) Rank CDF ----
sns.ecdfplot(
    data=combined_df,
    x='rank',
    hue='Approach',
    ax=axes[1]
)
axes[1].set_title('CDF of True Gene Rank')
axes[1].set_xlabel('Rank')

# ---- (3) Score vs Rank (Top10 highlighted) ----
for approach, sub in combined_df.groupby('Approach'):
    axes[2].scatter(
        sub[~sub['Top10_flag']]['rank'],
        sub[~sub['Top10_flag']]['score'],
        color=palette[approach],
        alpha=0.5,
        label=approach
    )
    axes[2].scatter(
        sub[sub['Top10_flag']]['rank'],
        sub[sub['Top10_flag']]['score'],
        color=palette[approach],
        edgecolor='black',
        s=90,
        label=f'{approach} Top10'
    )

axes[2].set_title('Score vs Rank (Top10 Highlighted)')
axes[2].set_xlabel('Rank')
axes[2].set_ylabel('Score')
axes[2].legend(fontsize=9)

# ---- (4) Top-N counts ----
sns.countplot(
    data=combined_df,
    x='TopN',
    hue='Approach',
    ax=axes[3]
)
axes[3].set_title('Top-N Recovery')
axes[3].set_xlabel('Top-N')

plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f'{OUT_PREFIX}_ranks.png', dpi=300)
plt.show()

# ==============================
# CONFUSION MATRIX BAR PLOT
# ==============================
conf_long = confusion_df.melt(
    id_vars='Approach',
    value_vars=['TP','FP','FN','TN'],
    var_name='Outcome',
    value_name='Count'
)

plt.figure(figsize=(11,6))
sns.barplot(
    data=conf_long,
    x='Approach',
    y='Count',
    hue='Outcome'
)
plt.title('Confusion Matrix Components by Approach')
plt.xticks(rotation=45)
plt.tight_layout()
if SAVE_FIGURES:
    plt.savefig(f'{OUT_PREFIX}_confusion.png', dpi=300)
plt.show()
