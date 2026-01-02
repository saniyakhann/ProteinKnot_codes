import csv
from collections import defaultdict

file = 'cath-domain-list.txt'

#dictionary to count domains per protein-chain
protein_chain_domains = defaultdict(list)

with open(file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            continue
        
        columns = line.strip().split()
        if len(columns) < 12:  
            continue
            
        domain_id = columns[0]
        
        #extract PDB ID and chain (first 5 characters: e.g., "1oaiA")
        pdb_chain = domain_id[:5]
        
        #store full CATH code (columns 1-9)
        cath_code = '.'.join(columns[1:10])
        
        n_residues = columns[10]  
        resolution = columns[11]  
        
        protein_chain_domains[pdb_chain].append({
            'domain_id': domain_id,
            'cath_code': cath_code,
            'n_residues': n_residues,
            'resolution': resolution
        })

single_domain_proteins = []

for pdb_chain, domains in protein_chain_domains.items():
    if len(domains) == 1:
        single_domain_proteins.append(domains[0])

print(f"Total protein-chain combinations: {len(protein_chain_domains)}")
print(f"Single-domain proteins: {len(single_domain_proteins)}")

with open('single_domain_proteins.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['domain_id', 'cath_code', 'pdb_id', 'chain', 'n_residues', 'resolution'])
    
    for protein in single_domain_proteins:
        domain_id = protein['domain_id']
        pdb_id = domain_id[:4]
        chain = domain_id[4]
        
        writer.writerow([
            domain_id,
            protein['cath_code'],
            pdb_id,
            chain,
            protein['n_residues'],
            protein['resolution']
        ])

print(f"Saved to single_domain_proteins.csv")