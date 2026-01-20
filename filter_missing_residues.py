from Bio import PDB
import requests
from io import StringIO
import pandas as pd
import time


df = pd.read_csv('single_domain_proteins.csv')
parser = PDB.PDBParser(QUIET=True)

clean_proteins = []
proteins_with_issues = []

for idx, row in df.iterrows():
    domain_id = row['domain_id']
    pdb_id = row['pdb_id']
    chain = row['chain']
    
    if (idx + 1) % 100 == 0:
        print(f"Progress: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%) - Clean: {len(clean_proteins)}, Issues: {len(proteins_with_issues)}")
    
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            proteins_with_issues.append(domain_id)
            continue
            
        pdb_io = StringIO(response.text)
        structure = parser.get_structure(pdb_id, pdb_io)
        
        residue_numbers = []
        model = structure[0]
        chain_upper = chain.upper()
        
        found_chain = False
        for pdb_chain in model:
            chain_id = pdb_chain.get_id()
            if chain_id.strip().upper() == chain_upper or (chain_id == ' ' and chain == '0'):
                for residue in pdb_chain:
                    if residue.has_id('CA'):
                        residue_numbers.append(residue.get_id()[1])
                found_chain = True
                break
        
        if not found_chain or len(residue_numbers) == 0:
            proteins_with_issues.append(domain_id)
            continue

      #filtering logic: 

        expected = residue_numbers[-1] - residue_numbers[0] + 1
        actual = len(residue_numbers)
        
        if expected == actual:
            clean_proteins.append(row)
        else:
            proteins_with_issues.append(domain_id)
            
    except Exception as e:
        proteins_with_issues.append(domain_id)
        continue
    
    time.sleep(0.1)

clean_df = pd.DataFrame(clean_proteins)
clean_df.to_csv('single_domain_proteins_complete.csv', index=False)

with open('proteins_excluded.txt', 'w') as f:
    for protein in proteins_with_issues:
        f.write(f"{protein}\n")

print(f"Original: {len(df)}")
print(f"No missing residues: {len(clean_proteins)}")
print(f"Excluded: {len(proteins_with_issues)}")