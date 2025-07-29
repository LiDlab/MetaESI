import tqdm
from Bio.PDB import PDBParser, DSSP
import pandas as pd

"""
Calculates Relative Solvent Accessibility (RSA) from AlphaFold2 PDB files.

Key Features:
- Uses DSSP to compute solvent accessibility from PDB structures
- Processes AlphaFold2 predictions for benchmark proteins
- Outputs both RSA and negative RSA (1-RSA) values

Workflow:
1. Loads list of benchmark protein IDs
2. For each protein:
   - Parses AlphaFold PDB file
   - Runs DSSP to calculate RSA values
   - Extracts position, amino acid, and RSA data
3. Computes negative RSA values (1-RSA)
4. Saves results to CSV file

Output:
- RSA.csv containing:
  - protein_id: UniProt ID
  - AA: Amino acid
  - position: Residue position  
  - RSA: Relative Solvent Accessibility (0-1)
  - neg_RSA: 1-RSA (inverse accessibility)

Dependencies:
- BioPython (PDBParser, DSSP)
"""

def extract_RSA_values(directory: str,
                       protein_ids: list):

    res = list()

    for protein_id in tqdm.tqdm(protein_ids):
        filepath = directory + "AF-" + protein_id +"-F1-model_v2.pdb"

        p = PDBParser()
        structure = p.get_structure("prot", filepath)
        model = structure[0]
        dssp = DSSP(model, filepath, dssp='dssp')

        # pos = [i[0] for i in dssp.property_list]
        pos = [i+1 for i in range(len(dssp.property_list))]
        AA = [i[1] for i in dssp.property_list]
        RSA = [i[3] for i in dssp.property_list]

        res_p = pd.DataFrame({'protein_id': protein_id,
                              'AA': AA,
                              'position': pos,
                              'RSA': RSA})

        res.append(res_p)

    res = pd.concat(res)

    return (res)


data_path = "../../data/human/raw/alphafold2/"
save_path = "../../results/idr_benchmark/"

benchmark = open(save_path + "benchmark_proteins.txt","r")
benchmark_proteins = benchmark.readlines()
benchmark_proteins = [x.strip() for x in benchmark_proteins]

RSA_data = extract_RSA_values(
    directory = data_path,
    protein_ids = benchmark_proteins)

RSA_data['neg_RSA'] = 1-RSA_data['RSA']
RSA_data.to_csv(save_path + 'RSA.csv', index=False)
