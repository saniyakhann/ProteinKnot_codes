import numpy as np
import os

def read_and_pad_matrix(npz_filepath, target_size):
    # Load matrix from .npz file
    data = np.load(npz_filepath)
    matrix = data['matrix']
    
    n = matrix.shape[0]
    
    if n < target_size:
        padding = target_size - n
        padded_matrix = np.pad(matrix, ((0, padding), (0, padding)), mode='constant', constant_values=0)
    elif n > target_size:
        padded_matrix = matrix[:target_size, :target_size]
    else:
        padded_matrix = matrix
    
    return padded_matrix

#using the maximum matrix size from data
input_dir = 'Writhe_Matrices_Clean'
output_dir = 'Writhe_Matrices_Padded'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]

max_size = 0
for file in files:
    filepath = os.path.join(input_dir, file)
    try:
        data = np.load(filepath)
        matrix = data['matrix']
        max_size = max(max_size, matrix.shape[0])
    except:
        continue

print(f"Maximum matrix size: {max_size}")

#pad all matrices to this target size
target_size = max_size  #rather than the 654 set manually
print(f"Padding all matrices to size: {target_size}x{target_size}")

for idx, file in enumerate(files):
    filepath = os.path.join(input_dir, file)
    
    try:
        padded_matrix = read_and_pad_matrix(filepath, target_size)

        #save
        output_filepath = os.path.join(output_dir, file)
        np.savez_compressed(output_filepath, matrix=padded_matrix)
        
        if (idx + 1) % 1000 == 0:
            print(f"Processed {idx + 1}/{len(files)} proteins")
    
    except Exception as e:
        print(f"Error processing {file}: {e}")
        continue

print(f"Padded matrices saved in: {output_dir}/")