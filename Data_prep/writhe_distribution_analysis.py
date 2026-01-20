import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def extract_total_writhe_per_protein(gauss_folder):
    total_writhes = []
    protein_names = []
    
    files = [f for f in os.listdir(gauss_folder) if f.endswith('.npz')]
    print(f"Found {len(files)} proteins")
    
    for idx, file in enumerate(files):
        filepath = os.path.join(gauss_folder, file)
        
        try:
            data = np.load(filepath)
            matrix = data['matrix']
            
           
            total_writhe = np.sum(np.triu(matrix, k=1))
            
            total_writhes.append(total_writhe)
            protein_names.append(file.replace('.npz', ''))
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1}/{len(files)} proteins")
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    print(f"\nExtracted total writhe for {len(total_writhes)} proteins")
    return np.array(total_writhes), protein_names

def plot_writhe_distribution(total_writhes):
    mean = np.mean(total_writhes)
    std = np.std(total_writhes)
    
    print(f"Total proteins: {len(total_writhes):,}")
    print(f"Mean:           {mean:+.6f}")
    print(f"Std deviation:  {std:.6f}")
    print(f"Min:            {np.min(total_writhes):+.6f}")
    print(f"Max:            {np.max(total_writhes):+.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(total_writhes, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, label='Zero')
    plt.xlabel('Total Writhe Value', fontsize=13)
    plt.ylabel('Number of Proteins', fontsize=13)
    plt.title('Distribution of Total Writhe per Protein', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('writhe_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

total_writhes, protein_names = extract_total_writhe_per_protein('Writhe_Matrices_Clean')
plot_writhe_distribution(total_writhes)
