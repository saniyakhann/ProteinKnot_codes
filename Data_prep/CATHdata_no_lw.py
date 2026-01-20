from Bio import PDB
import requests
from io import StringIO
import numpy as np
from gauss_updated import compute_writhe_matrix  
import csv
import os
import random

# Read the clean single-domain proteins file 
input_file = 'single_domain_proteins.csv'
proteins = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        proteins.append(row)
proteins = random.sample(proteins, 1000) 

print(f"Loading {len(proteins)} clean single-domain proteins...")

broken_proteins = []
parser = PDB.PDBParser(QUIET=True)

# Create output directory if it doesn't exist
output_dir = 'Writhe_Matrices_Clean'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for idx, data in enumerate(proteins):
    domain_id = data['domain_id']
    pdb_id = data['pdb_id']
    chain = data['chain']
    
    print(f"Processing protein {idx+1}/{len(proteins)}: {domain_id} ({pdb_id}-{chain})")
    
    file_path = f'{output_dir}/{domain_id}.npz'
    
    # Skip if already processed
    if os.path.exists(file_path):
        print(f"  File {file_path} already exists, skipping...")
        continue
    
    # Download PDB structure
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"  Failed to download {pdb_id}")
        broken_proteins.append(domain_id)
        continue
        
    pdb_data = response.text
    pdb_io = StringIO(pdb_data)
    structure = parser.get_structure(f'{pdb_id}', pdb_io)
    
    # Extract CA atoms from the specific chain
    carbon_atoms = []
    model = structure[0]
    chain_upper = chain.upper()
    
    for pdb_chain in model:
        chain_id = pdb_chain.get_id()
        
        if chain_id.strip().upper() == chain_upper or (chain_id == ' ' and chain == '0'):
            for residue in pdb_chain:
                if residue.has_id('CA'):
                    carbon_atoms.append(residue['CA'].coord)
            break
                    
    carbon_atoms = np.array(carbon_atoms)
    
    if len(carbon_atoms) == 0:
        print(f"  No CA atoms found in chain {chain} of protein {pdb_id}")
        broken_proteins.append(domain_id)
        continue
        
    print(f"  Found {len(carbon_atoms)} CA atoms, computing writhe matrix...")
    
    # Compute writhe matrix
    matrix = compute_writhe_matrix(carbon_atoms, carbon_atoms)
    
    # Calculate and print total writhe
    total_writhe = np.sum(matrix)
    print(f"  Total writhe: {total_writhe:.3f}")
    
    # Save matrix
    matrix = np.round(matrix, 5)
    np.savez_compressed(file_path, matrix=matrix)
    print(f"  Progress: {((idx+1)/len(proteins))*100:.1f}%")

# Save broken proteins
with open('broken_proteins_clean.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['domain_id'])
    for protein in broken_proteins:
        writer.writerow([protein])

print(f"\n=== Processing Complete! ===")
print(f"Successful: {len(proteins) - len(broken_proteins)}")
print(f"Failed: {len(broken_proteins)}")
print(f"Writhe matrices saved in: {output_dir}/")
