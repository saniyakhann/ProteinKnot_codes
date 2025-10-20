##only a few changes made from Djordje's original code


from Bio import PDB
import requests
from io import StringIO
import numpy as np
from gauss import *  
import csv
import os

lp = 2
protein_lengths = []
proteins = []
# file = '/Users/s1910360/Desktop/ML for Knot Theory/Protein data/cath-chain-list.txt' # original line from Djorjde's code 
file = 'cath-chain-list.txt'  # local file reading 

with open(file, newline='') as f:
    reader = csv.reader(f)
    for row in f:
        if row.startswith('#'):
            continue
        columns = row.strip().split()
        proteins.append(columns)
        if len(proteins) >= 1000:  # limited to 1000 protein data 
            break

print(f"Loading {len(proteins)} proteins...")

broken_proteins = []
parser = PDB.PDBParser(QUIET=True)

#create output directory if it doesn't exist
if not os.path.exists('Protein Gauss'):
    os.makedirs('Protein Gauss')

for idx, data in enumerate(proteins):
    prot = data[0][:4]
    chain_char = data[0][4:5]
    domain = data[0][6:]

    print(f"Processing protein {idx+1}/{len(proteins)}: {prot}-{chain_char}")
    file_path = f'Protein Gauss/{prot}-{chain_char}.npz'
    
    if os.path.exists(file_path):
        print(f"File {file_path} already exists, skipping...")
        continue

    url = f'https://files.rcsb.org/download/{prot}.pdb'
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Failed to download {prot}")
        broken_proteins.append(f"{prot}-{chain_char}")
        continue
        
    pdb_data = response.text
    pdb_io = StringIO(pdb_data)
    structure = parser.get_structure(f'{prot}', pdb_io)

    carbon_atoms = []
    last_residue_id = None
    model = structure[0]

    for chain in model:
        chain_id = chain.get_id()
        
        if chain_id[0] == chain_char or (chain_id == ' ' and chain_char == '0'):
            for residue in chain:
                if residue.has_id('CA'):
                    current_residue_id = residue.get_id()[1]
                    
                    if last_residue_id is None or current_residue_id == last_residue_id + 1:
                        carbon_atoms.append(residue['CA'].coord)
                    last_residue_id = current_residue_id
                    
    carbon_atoms = np.array(carbon_atoms)

    if len(carbon_atoms) == 0:
        print(f"No carbon atoms found in chain {chain_char} of protein {prot}")
        broken_proteins.append(f"{prot}-{chain_char}")
        continue
        
    print(f"  Found {len(carbon_atoms)} CA atoms, computing writhe matrix...")
    matrix = compute_sts_writhe(carbon_atoms, carbon_atoms, lp)
    matrix = np.round(matrix, 5)
    np.savez_compressed(file_path, matrix=matrix)
    print(f"  Progress: {((idx+1)/len(proteins))*100:.1f}%")

#save broken proteins
with open('broken.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['protein_chain'])
    for protein in broken_proteins:
        writer.writerow([protein])

print(f"\nProcessing complete!")
print(f"Successful: {len(proteins) - len(broken_proteins)}")
print(f"Failed: {len(broken_proteins)}")