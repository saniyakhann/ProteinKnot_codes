import pandas as pd

df = pd.read_csv('single_domain_proteins_complete.csv')

def CATH_value_counts(df, cath_col='cath_code'):
    total = len(df) 
    print(f"Total proteins: {total:,}\n")
    
    #extract CATH levels as integers
    cath_levels = {
        'C': df[cath_col].astype(str).apply(lambda x: int(x.split('.')[0])),
        'A': df[cath_col].astype(str).apply(lambda x: int(x.split('.')[1])),
        'T': df[cath_col].astype(str).apply(lambda x: int(x.split('.')[2])),
        'H': df[cath_col].astype(str).apply(lambda x: int(x.split('.')[3]))
    }
    
    print("Summary")
    for level_name, level_series in cath_levels.items():
        n_unique = level_series.nunique()
        print(f"{level_name}: {n_unique} distinct values")
    
    print("\nValue Distributions")
    for level_name, level_series in cath_levels.items():
        counts = level_series.value_counts().sort_index()
        print(f"\n{level_name} Level:")
        for value, count in counts.items():
            percentage = (count / total) * 100
            print(f"  {level_name}={value:>3d}: {count:6d} proteins ({percentage:.2f}%)")

CATH_value_counts(df)