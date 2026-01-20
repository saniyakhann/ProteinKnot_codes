import numpy as np
import pandas as pd
import os

#paths
matrices_dir = 'Writhe_Matrices_Padded'
csv_path = 'single_domain_proteins_complete.csv'

#load CSV and then extracting C-level class
df = pd.read_csv(csv_path)
df['c_level'] = df['cath_code'].apply(lambda x: int(str(x).split('.')[0]))

#get matrix files
matrix_files = [f for f in os.listdir(matrices_dir) if f.endswith('.npz')]
matrix_ids = [f.replace('.npz', '') for f in matrix_files]

#matching with the labels
df_matched = df[df['domain_id'].isin(matrix_ids)]

print(f"Total matrices: {len(matrix_files)}")
print(f"Matched with labels: {len(df_matched)}")
print(f"\nClass distribution:")
print(df_matched['c_level'].value_counts().sort_index())

#check one matrix
sample = np.load(os.path.join(matrices_dir, matrix_files[0]))['matrix']
print(f"\nMatrix shape: {sample.shape}")
print(f"Value range: [{sample.min():.3f}, {sample.max():.3f}]")