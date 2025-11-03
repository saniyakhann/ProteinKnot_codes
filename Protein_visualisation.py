import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import requests
import pandas as pd
import warnings

def parse_cath_file(file_path="cath-domain-list.txt"):
   rows = []
   with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 12:
                domain_id = parts[0]               #
                pdb_id = domain_id[:4].lower()     
                chain_id = domain_id[4].upper()    
                domain_no = domain_id[5:7]         
                
                cath_class = parts[1]
                cath_arch  = parts[2]
                cath_topo  = parts[3]
                cath_homo  = parts[4]
                length = int(parts[-2])            
                resolution = float(parts[-1])      
                rows.append({
                    "domain_id": domain_id,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "domain_no": domain_no,
                    "cath_code": ".".join([cath_class, cath_arch, cath_topo, cath_homo]),
                    "cath_class": cath_class,
                    "residue_count": length,
                    "resolution": resolution
                })
   return pd.DataFrame(rows)

df = parse_cath_file("cath-domain-list.txt")

def get_ca_coordinates(pdb_id, chain_id='A'):
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to download {pdb_id}")
            return None
        
        ca_coords = []
        residue_numbers = []
        
        for line in response.text.split('\n'):
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                chain = line[21:22].strip()
                
                if atom_name == 'CA' and (chain == chain_id or (chain == '' and chain_id == '0')):
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        res_num = int(line[22:26])
                        ca_coords.append([x, y, z])
                        residue_numbers.append(res_num)
                    except:
                        continue
        
        if ca_coords:
            return np.array(ca_coords), residue_numbers
        return None
        
    except Exception as e: 
        print(f"Error processing {pdb_id}: {e}")
        return None
    
def visualise_domain(domain_row, figsize=(15,5), residue_ranges=None):
    pdb_id = domain_row['pdb_id']
    chain_id = domain_row['chain_id']
    
    result = get_ca_coordinates(pdb_id, chain_id) 
    coords, res_numbers = result
    
    #create figure with 3 subplots
    fig = plt.figure(figsize=figsize)
    
    # 1. 3D structure
    ax1 = fig.add_subplot(131, projection='3d')
    
    # plot backbone
    ax1.plot(coords[:, 0], coords[:, 1], coords[:, 2], 
             'b-', linewidth=1.5, alpha=0.6)
    
    #color points by position
    colors = np.arange(len(coords))
    scatter = ax1.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                         c=colors, cmap='rainbow', s=20, alpha=0.8)
    
    ax1.set_xlabel('X (Å)', fontsize=10)
    ax1.set_ylabel('Y (Å)', fontsize=10)
    ax1.set_zlabel('Z (Å)', fontsize=10)
    ax1.set_title(f'{pdb_id}-{chain_id} 3D Structure\n{len(coords)} residues', fontsize=12)
    
    # 2. distance matrix
    ax2 = fig.add_subplot(132)
    
    # calculate pairwise distances
    dist_matrix = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
    
    im = ax2.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax2.set_xlabel('Residue index', fontsize=10)
    ax2.set_ylabel('Residue index', fontsize=10)
    ax2.set_title('Distance Matrix (Å)', fontsize=12)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    
    # 3. projections
    ax3 = fig.add_subplot(133)
    
    # XY projection with color gradient
    scatter2 = ax3.scatter(coords[:, 0], coords[:, 1], 
                          c=colors, cmap='rainbow', s=15, alpha=0.7)
    ax3.plot(coords[:, 0], coords[:, 1], 'gray', linewidth=0.5, alpha=0.3)
    
    ax3.set_xlabel('X (Å)', fontsize=10)
    ax3.set_ylabel('Y (Å)', fontsize=10)
    ax3.set_title('XY Projection', fontsize=12)
    ax3.set_aspect('equal')
    plt.colorbar(scatter2, ax=ax3, fraction=0.046, pad=0.04, label='Residue index')
    
    plt.suptitle(f'Protein Structure Analysis: {pdb_id} Chain {chain_id}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return coords, dist_matrix

#user inputs protein ID
protein_input = input(f"\nEnter protein ID: ").strip()
if protein_input.endswith('.npz'):
    protein_input = protein_input[:-4]

#find protein in dataframe using the full domain ID
matching_proteins = df[df['domain_id'] == protein_input]

if matching_proteins.empty:
    print(f"\nError: No protein found with domain ID '{protein_input}'")
    print(f"Make sure you entered the correct domain ID from the CATH file.")
    print(f"Example from your file: 1oaiA00")
    exit()

protein = matching_proteins.iloc[0]

print(f"Domain: {protein['domain_id']}")
print(f"PDB: {protein['pdb_id']}  Chain: {protein['chain_id']}  DomainNo: {protein['domain_no']}")
print(f"CATH: {protein['cath_code']}  (Class {protein['cath_class']})")
print(f"Domain residues (count): ~{protein['residue_count']}  Resolution: {protein['resolution']} Å")
coords, dist_matrix = visualise_domain(protein)