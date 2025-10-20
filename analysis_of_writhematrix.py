import numpy as np
import os

writhe_matrix = 'Protein Gauss'
npz_files = [f for f in os.listdir(writhe_matrix) if f.endswith('.npz')]

print(f"Total number of protein files: {len(npz_files)}")

nan_files = []
non_symmetric_files = []
clean_files = []

for idx, file in enumerate(npz_files):
    filepath = os.path.join(writhe_matrix, file)
    data = np.load(filepath)
    matrix = data['matrix']
    
    nan_mask = np.isnan(matrix) #numpy function that detects NaN values 
    nan_count = np.sum(nan_mask)
    has_nan = nan_count > 0
    nan_percentage = (nan_count / matrix.size) * 100
    
    #max to find max deviation from symmetry 
    symmetry_diff = np.max(np.abs(matrix - matrix.T)) #transpose matrix (rows are columns and columns are rows)
    symmetric = symmetry_diff < 1e-10
    
    if has_nan:
        #finding where the NaNs are
        nan_indices = np.where(nan_mask)  #numpy function that finds positions
        nan_on_diagonal = np.sum([i == j for i, j in zip(nan_indices[0], nan_indices[1])]) #row indices and column indices where Nans where found 
        nan_off_diagonal = nan_count - nan_on_diagonal
        
        nan_files.append((file, nan_count, nan_percentage, nan_on_diagonal, nan_off_diagonal))
        print(f"{file}: {nan_count} NaNs ({nan_on_diagonal} on diagonal, {nan_off_diagonal} off diagonal)")
        
    elif not symmetric:
        #show asymmetry values
        non_symmetric_files.append((file, symmetry_diff))
        print(f"{file}: Asymmetric (max diff: {symmetry_diff:.10f})")
        
    else:
        clean_files.append(file)
        print(f"{file}: Clean")

print("Results:")
print(f"Clean files: {len(clean_files)}")
print(f"Files with NaN values: {len(nan_files)}")
print(f"Non-symmetric files: {len(non_symmetric_files)}")

if nan_files:
    print(f"\nNaN Patterns:")
    #show files with the most NaN problems
    worst_nan = sorted(nan_files, key=lambda x: x[2], reverse=True)[:10]
    for file, count, percentage, on_diag, off_diag in worst_nan:
        print(f"  {file}: {percentage:.1f}% NaN ({on_diag} on diag, {off_diag} off diag)")

if non_symmetric_files:
    print(f"\nAsymmetry Patterns:")
    #show range of actual asymmetry values
    asymmetries = [diff for _, diff in non_symmetric_files] #keeping diff as the 'amount' of asymmetry 
    print(f"  Asymmetry range: {min(asymmetries):.10f} to {max(asymmetries):.10f}") #smallest to largest asymmetry value found 
    print(f"  Mean asymmetry: {np.mean(asymmetries):.10f}")
    
    #show most asymmetric files
    worst_asym = sorted(non_symmetric_files, key=lambda x: x[1], reverse=True)[:10]
    for file, diff in worst_asym:
        print(f"  {file}: {diff:.10f}")

print(f"\n% of usable files: {len(clean_files)/len(npz_files)*100:.1f}%")