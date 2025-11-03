import pandas as pd

def create_cath_protein_list(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:

            #skip comment lines
            if line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) < 5:
                continue
            
            domain_id = parts[0]  # e.g: 1oaiA00
            
            #extract the protein ID
            protein_id = domain_id
            
            #extract CATH 
            cath_class = parts[1]         # C
            cath_arch = parts[2]          # A
            cath_topo = parts[3]          # T
            cath_homol = parts[4]         # H
            
            cath_code = f"{cath_class}.{cath_arch}.{cath_topo}.{cath_homol}"
            
            data.append({
                'protein_id': protein_id,
                'cath_code': cath_code
            })
    
    return pd.DataFrame(data)

cath_proteins_df = create_cath_protein_list('cath-domain-list.txt')
output_file = 'CATH_proteins.csv'
cath_proteins_df.to_csv(output_file, index=False)
