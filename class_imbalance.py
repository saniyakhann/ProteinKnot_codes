#percentage of proteins at each level:
    #C (Class): first number
    #A (Architecture): first two numbers 
    #T (Topology): first three numbers
    #H (Homology): the full code 
import pandas as pd
import os
import matplotlib.pyplot as plt

df = pd.read_csv('single_domain_proteins.csv')
writhe_dir = 'Writhe_Matrices_Clean'

calculated_proteins = []
for idx, row in df.iterrows():
    domain_id = row['domain_id']
    npz_file = f'{writhe_dir}/{domain_id}.npz'
    if os.path.exists(npz_file):
        calculated_proteins.append(row)

df = pd.DataFrame(calculated_proteins)

def CATH_imbalance(df, cath_col='cath_code'):
    total = len(df) 
    print(f"Total proteins: {total:,}")
    
    cath_levels = {
        'C': df[cath_col].astype(str).apply(lambda x: x.split('.')[0]),
        'A': df[cath_col].astype(str).apply(lambda x: '.'.join(x.split('.')[:2])),
        'T': df[cath_col].astype(str).apply(lambda x: '.'.join(x.split('.')[:3])),
        'H': df[cath_col].astype(str)
    }
    print("Summary:")
    for level_name, level_series in cath_levels.items():
        n_unique = level_series.nunique()
        print(f"{level_name}: {n_unique:>4} unique codes")
    print()
    
    for level_name, level_series in cath_levels.items():
        percentages = level_series.value_counts().sort_index() / total * 100
        
        print(f"{level_name} Distribution:")
        for code, percentage in percentages.items():
            print(f"{code:15s}: {percentage:.10}%")
        print(f"Unique {level_name} codes: {len(percentages)}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, (level_name, level_series) in enumerate(cath_levels.items()):
        ax = axes[idx // 2, idx % 2]
        counts = level_series.value_counts().sort_values(ascending=False).head(20)
        counts.plot(kind='bar', ax=ax)
        ax.set_title(f'{level_name} Level')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
    plt.tight_layout()
    plt.savefig('CATH_distribution.png', dpi=300)
    plt.show()
    
    print("\nSummary:")
    for level_name, level_series in cath_levels.items():
        print(f"{level_name}: {level_series.nunique()} unique codes")

CATH_imbalance(df)