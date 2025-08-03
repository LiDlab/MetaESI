import os
from Bio import PDB
from tqdm import tqdm

"""
Extracts protein sequences from AlphaFold2 PDB files for ESM-2 model feature embedding.

Key Functions:
- write_fasta(): Writes protein sequences to FASTA format file
- Main processing loop:
  1. Scans AlphaFold2 PDB directory
  2. Parses each structure using BioPython
  3. Extracts and concatenates sequences from all chains
  4. Stores sequences with UniProt IDs

Workflow:
1. Identifies all AlphaFold2 PDB files (*-F1-model_v2.pdb)
2. For each file:
   - Parses structure using PDBParser
   - Builds complete sequence including all chains
   - Maps sequence to UniProt ID (from filename)
3. Outputs all sequences to FASTA file

Output:
- MetaESI_seq.fasta: Contains protein sequences in format:
  >UniProtID
  SEQUENCE

Dependencies:
- BioPython (PDB parser)
"""

script_dir = os.path.dirname(os.path.abspath(__file__))
relative_path1 = "../../data/human/features/"
relative_path2 = "../../data/human/raw/alphafold2/"

save_path = os.path.join(script_dir, relative_path1)
af2_path = os.path.join(script_dir, relative_path2)

os.makedirs(save_path, exist_ok=True)

pdb_files = [filename for filename in os.listdir(af2_path) if filename.endswith("-F1-model_v2.pdb")]

protein_sequences = {}
parser = PDB.PDBParser()
for filename in tqdm(pdb_files, desc="Processing PDB files"):
    if filename.endswith("-F1-model_v2.pdb"):
        structure = parser.get_structure(filename, os.path.join(af2_path, filename))
        sequence = ''
        for model in structure:
            for chain in model:
                polypeptides = PDB.PPBuilder().build_peptides(chain)
                for poly_index, poly in enumerate(polypeptides):
                    sequence += str(poly.get_sequence())
        protein_sequences[filename.split('-')[1]] = sequence


def write_fasta(protein_sequences, output_file):
    with open(output_file, 'w') as f:
        for filename, sequence in protein_sequences.items():
            f.write(f">{filename}\n")
            f.write(sequence + "\n")
write_fasta(protein_sequences, save_path + "MetaESI_seq.fasta")
