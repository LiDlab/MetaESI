from pPSE_utils import format_alphafold_data, annotate_accessibility

"""
Processes AlphaFold2 predictions to calculate pPSE (prediction-aware part-sphere exposure) scores.

Workflow:
1. Loads benchmark protein IDs from text file
2. Processes AlphaFold2 PDB files to extract structural annotations:
   - format_alphafold_data(): Extracts basic structural features
   - annotate_accessibility(): Calculates surface accessibility metrics
3. Saves two output files:
   - alphafold_annotation.csv: Raw structural annotations
   - pPSE.csv: Final surface exposure predictions

Parameters:
- max_dist: Maximum distance cutoff for accessibility calculation (24Å)
- max_angle: Angular cutoff for exposure calculation (180°)
- error_dir: Path to AlphaFold2 PAE files

Dependencies:
- pPSE_utils: Contains the core processing functions
"""

data_path = "../../data/human/raw/alphafold2/"
save_path = "../../results/idr_benchmark/"

benchmark = open(save_path + "benchmark_proteins.txt","r")
benchmark_proteins = benchmark.readlines()
benchmark_proteins = [x.strip() for x in benchmark_proteins]

# pPSE
alphafold_annotation = format_alphafold_data(directory=data_path,
                                             protein_ids=benchmark_proteins)

alphafold_annotation.to_csv(save_path + "alphafold_annotation.csv",index=False)

acc_res = annotate_accessibility(df=alphafold_annotation,
                                 max_dist=24,
                                 max_angle=180,
                                 error_dir=data_path)

acc_res.to_csv(save_path + 'pPSE.csv', index=False)