#to calculate the distance matrices 
from Bio import PDB
import requests
from io import StringIO
import numpy as np
import csv
import os

input_file = 'proteins_final.csv'
proteins = []

with open(input_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        proteins.append(row)

parser = PDB.PDBParser(QUIET=True)

output_dir = 'Distance_Matrices_256'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

broken_proteins = []

def compute_distance_matrix(coords):
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    matrix = np.sqrt((diff ** 2).sum(axis=-1))
    return matrix

def downsample_block_mean(mat, out=256):
    H, W = mat.shape
    newH = int(np.ceil(H / out) * out)
    newW = int(np.ceil(W / out) * out)
    padH = newH - H
    padW = newW - W
    if padH or padW:
        mat = np.pad(mat, ((0, padH), (0, padW)), mode="constant", constant_values=0.0)
    mat = mat.reshape(out, newH // out, out, newW // out).mean(axis=(1, 3))
    return mat

for idx, data in enumerate(proteins):
    domain_id = data['domain_id']
    pdb_id = data['pdb_id']
    chain = data['chain']

    file_path = f'{output_dir}/{domain_id}.npz'

    if os.path.exists(file_path):
        continue

    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    response = requests.get(url)

    if response.status_code != 200:
        broken_proteins.append(domain_id)
        continue

    pdb_data = response.text
    pdb_io = StringIO(pdb_data)
    structure = parser.get_structure(f'{pdb_id}', pdb_io)

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
        broken_proteins.append(domain_id)
        continue

    matrix = compute_distance_matrix(carbon_atoms)
    matrix = downsample_block_mean(matrix, out=256)
    matrix = np.round(matrix, 5)
    np.savez_compressed(file_path, matrix=matrix)

with open('broken_proteins_distance.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['domain_id'])
    for protein in broken_proteins:
        writer.writerow([protein])
