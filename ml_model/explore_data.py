import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

#paths
matrices_dir = 'Writhe_Matrices_Padded'
csv_path = 'single_domain_proteins_complete.csv'

#load CSV
df = pd.read_csv(csv_path)

#df['c_level'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))
#all levels:
df['C'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))
df['A'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[1]))
df['T'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[2]))
df['H'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[3]))

#get matrix files
matrix_files = [f for f in os.listdir(matrices_dir) if f.endswith('.npz')]
matrix_ids = [f.replace('.npz', '') for f in matrix_files]

#matching with the labels
df_matched = df[df['domain_id'].isin(matrix_ids)]

print(f"Total matrices: {len(matrix_files)}")
print(f"Matched with labels: {len(df_matched)}")
levels = ['C', 'A', 'T', 'H']
level_names = ['Class', 'Architecture', 'Topology', 'Homology']

for level, name in zip(levels, level_names):
    print(f"\n{name} ({level})-level distribution:")
    print(df_matched[level].value_counts().sort_index())

#check one matrix
sample = np.load(os.path.join(matrices_dir, matrix_files[0]))['matrix']
print(f"\nMatrix shape: {sample.shape}")
print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")

#plot distributions for all levels
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, (level, name) in enumerate(zip(levels, level_names)):
    ax = axes[idx // 2, idx % 2]
    counts = df_matched[level].value_counts().sort_index()
    
    ax.bar(counts.index.astype(str), counts.values, 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel(f'{level} Value', fontsize=12)
    ax.set_ylabel('Number of Proteins', fontsize=12)
    ax.set_title(f'{name} ({level})-level Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    if len(counts) > 10:
        ax.tick_params(axis='x', rotation=45)
    
    if len(counts) <= 20:
        for i, (val_idx, val) in enumerate(counts.items()):
            ax.text(i, val + max(counts.values) * 0.02, str(val), 
                   ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('cath_all_levels_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
