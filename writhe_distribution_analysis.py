import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def writhe_values_extraction(gauss_folder, max_proteins=1000):
    all_writhe_values = []
    protein_count = 0
    
    for file in os.listdir(gauss_folder):
        if file.endswith('.npz') and protein_count < max_proteins:
            filepath = os.path.join(gauss_folder, file)
            
            try:
                data = np.load(filepath)
                matrix = data['matrix']
                
                #upper triangle (excluding diagonal)
                upper_triangle_indices = np.triu_indices_from(matrix, k=1)
                writhe_values = matrix[upper_triangle_indices]
                
                all_writhe_values.extend(writhe_values)
                protein_count += 1
    
                if protein_count % 100 == 0:
                    print(f"  Processed {protein_count} proteins...")
            
            except Exception as e:
                continue
    
    print(f"\nExtracted {len(all_writhe_values):,} writhe values from {protein_count} proteins")
    
    return np.array(all_writhe_values)


def plot_writhe_distribution(writhe_values):
    #calculate statistics
    mean = np.mean(writhe_values)
    std = np.std(writhe_values)


    print(f"Total values:  {len(writhe_values):,}")
    print(f"Mean:          {mean:+.6f}")
    print(f"Std deviation: {std:.6f}")
    print(f"Min:           {np.min(writhe_values):+.6f}")
    print(f"Max:           {np.max(writhe_values):+.6f}")
    print()
    
    #Plotting:

    plt.figure(figsize=(10, 6))
    
    #histogram
    plt.hist(writhe_values, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    
    #mark mean and zero
    plt.axvline(mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean:.4f}')
    plt.axvline(0, color='black', linestyle='-', linewidth=1.5, 
                label='Zero')
    
    #labels
    plt.xlabel('Writhe Value', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.title('Distribution of Writhe Values', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('writhe_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
    #check if peaked in middle
    if abs(mean) < 0.01:
        print("\nPeak is centered near zero")
    else:
        print(f"\nPeak is offset from zero by {mean:.4f}")

#extract writhe values
writhe_values = writhe_values_extraction('Protein Gauss', max_proteins=1000)

plot_writhe_distribution(writhe_values)