import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

writhe = pd.read_csv('gradcam_results_writhe_A.csv')

a_classes = sorted(writhe['true_A'].unique())
positions = list(range(1, len(a_classes) + 1))
colors = plt.cm.tab20(np.linspace(0, 1, len(a_classes)))

fig, ax = plt.subplots(figsize=(14, 6))

for i, a_val in enumerate(a_classes):
    dar = writhe[writhe['true_A'] == a_val]['DAR'].values
    if len(dar) < 3:
        continue
    ax.boxplot(dar, positions=[positions[i]], widths=0.6,
               patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=colors[i], alpha=0.7),
               medianprops=dict(color='black', linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5))
    correct = writhe[(writhe['true_A'] == a_val) & (writhe['correct'] == 1)]['DAR'].values
    print(f'A{a_val}: n={len(dar)}, median={np.median(dar):.3f}, correct_median={np.median(correct):.3f}')

ax.set_xticks(positions)
ax.set_xticklabels([f'A{a}' for a in a_classes], fontsize=8, rotation=45)
ax.set_ylabel('Diagonal Attention Ratio (DAR)', fontsize=12)
ax.set_title('Grad-CAM Diagonal Attention Ratio by Architecture Class\n(Writhe CNN, A-level)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('DAR_boxplot_writhe_only_A.png', dpi=200, bbox_inches='tight')
plt.close()
print('saved: DAR_boxplot_writhe_only_A.png')
