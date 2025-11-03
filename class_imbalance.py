#percentage of proteins at each level:
    #C (Class): first number
    #A (Architecture): first two numbers 
    #T (Topology): first three numbers
    #H (Homology): the full code 

import pandas as pd

df = pd.read_csv('CATH_proteins.csv')

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

CATH_imbalance(df)