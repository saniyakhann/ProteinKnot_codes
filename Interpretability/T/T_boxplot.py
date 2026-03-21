import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

writhe = pd.read_csv('gradcam_results_writhe_T.csv')

class_order = [1, 2, 3, 4]
class_names = ['Mainly Alpha', 'Mainly Beta', 'Alpha/Beta', 'Few SS']
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

fig, ax = plt.subplots(figsize=(8, 6))

for i, (cls_idx, cls_name) in enumerate(zip(class_order, class_names)):
    dar = writhe[writhe['C_val'] == cls_idx]['DAR'].values
    if len(dar) < 3:
        continue
    ax.boxplot(dar, positions=[i + 1], widths=0.6,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=colors[i], alpha=0.7),
               medianprops=dict(color='black', linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    median = np.median(dar)
    ax.text(i + 1, median + 0.02, f'median={median:.3f}', ha='center', fontsize=9, fontweight='bold')
    correct = writhe[(writhe['C_val'] == cls_idx) & (writhe['correct'] == 1)]['DAR'].values
    wrong = writhe[(writhe['C_val'] == cls_idx) & (writhe['correct'] == 0)]['DAR'].values
    wrong_median = np.median(wrong) if len(wrong) > 0 else float('nan')
    print(f'{cls_name}: n={len(dar)}, median={median:.3f}, correct_median={np.median(correct):.3f}, wrong_median={wrong_median:.3f}')

alpha_dar = writhe[writhe['C_val'] == 1]['DAR'].values
beta_dar = writhe[writhe['C_val'] == 2]['DAR'].values
_, p = stats.mannwhitneyu(alpha_dar, beta_dar, alternative='two-sided')
print(f'mann-whitney u (alpha vs beta): p={p:.4e}')

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(class_names, fontsize=11)
ax.set_ylabel('Diagonal Attention Ratio (DAR)', fontsize=12)
ax.set_title('Grad-CAM Diagonal Attention Ratio by CATH Class\n(Writhe CNN, T-level)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('DAR_boxplot_writhe_only_T.png', dpi=200, bbox_inches='tight')
plt.close()
print('saved: DAR_boxplot_writhe_only_T.png')
