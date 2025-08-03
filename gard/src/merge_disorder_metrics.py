from pPSE_utils import get_smooth_score
import pandas as pd
import numpy as np

"""
Integrates and processes multiple disorder prediction metrics into a unified benchmark dataset.

Key Processing Steps:
1. Loads and merges data from multiple sources:
   - AlphaFold structural annotations (pLDDT)
   - Surface accessibility (RSA)
   - Disorder predictions (IUPred3)
   - Graph-aware Residue Depth (GARD)
   - Prediction-aware part-sphere exposure (pPSE)
   - Experimental ground truth labels

2. Data Cleaning:
   - Identifies and removes proteins with sequence mismatches
   - Applies smoothing functions to selected metrics
   - Normalizes values to consistent scales

3. Output Preparation:
   - Renames columns for clarity
   - Normalizes pPSE and pLDDT scores
   - Saves final integrated dataset

Output Features:
- protein_id: UniProt accession
- AA: Amino acid
- position: Residue position
- pLDDT: Confidence score (0-1)
- RSA: Relative solvent accessibility (0-1)
- IUPred3: Disorder probability (0-1) 
- pPSE: Prediction-aware part-sphere exposure (normalized)
- GARD: Graph-aware residue depth score
- label: Ground truth (1=ordered, 0=disordered)

Dependencies:
- pPSE_utils for smoothing functions
"""

data_path = "../../results/idr_benchmark/"

benchmark = open(data_path + "benchmark_proteins.txt","r")
benchmark_proteins = benchmark.readlines()
benchmark_proteins = [x.strip() for x in benchmark_proteins]

alphafold_annotation = pd.read_csv(data_path + "alphafold_annotation.csv")

# pPSE
pPSE = pd.read_csv(data_path + "pPSE.csv")
benchmark = alphafold_annotation.merge(pPSE, how='left', on=['protein_id','AA','position'])

# RSA
RSA = pd.read_csv(data_path + "RSA.csv")
benchmark = benchmark.merge(RSA, how='left', on=['protein_id','AA','position'])

# iupred3
iupred3 = pd.read_csv(data_path + "iupred3.csv")
benchmark = benchmark.merge(iupred3, how='left', on=['protein_id','AA','position'])

# RD_graph
RD_graph = pd.read_csv(data_path + "GARD.csv")
benchmark = benchmark.merge(RD_graph, how='left', on=['protein_id','position'])

invalid_proteins = list(benchmark[np.isnan(benchmark.neg_iupred3)].protein_id.unique())
print("Invalid proteins: ", invalid_proteins)
benchmark = benchmark[-benchmark.protein_id.isin(invalid_proteins)].reset_index(drop=True)

# smooth
benchmark_smooth = get_smooth_score(
    benchmark,
    np.array(['quality',
              'neg_RSA',
              'nAA_24_180_pae',]),[10,20])

# Merge computed data with groudtruth data
groundtruth_data = pd.read_csv(data_path + "groundtruth_data.csv")
alphafold_disorder_benchmark = benchmark_smooth.merge(groundtruth_data, how='left',on=['protein_id','position'])
alphafold_disorder_benchmark_sub = alphafold_disorder_benchmark[(alphafold_disorder_benchmark.ordered==1) | (alphafold_disorder_benchmark.disordered==1)]

alphafold_disorder_benchmark_sub = alphafold_disorder_benchmark_sub[["protein_id", "AA", "position", "quality_smooth20", "neg_RSA_smooth20", "neg_iupred3", "nAA_24_180_pae_smooth10", "GARD", "disordered", "ordered"]]
alphafold_disorder_benchmark_sub["quality_smooth20"] = alphafold_disorder_benchmark_sub["quality_smooth20"]/100
alphafold_disorder_benchmark_sub["nAA_24_180_pae_smooth10"] = alphafold_disorder_benchmark_sub["nAA_24_180_pae_smooth10"]/np.max(alphafold_disorder_benchmark_sub["nAA_24_180_pae_smooth10"].values)

alphafold_disorder_benchmark_sub = alphafold_disorder_benchmark_sub.rename(columns={
    'quality_smooth20': 'pLDDT',
    'neg_RSA_smooth20': 'RSA',
    'neg_iupred3': 'IUPred3',
    'nAA_24_180_pae_smooth10': 'pPSE',
    'ordered': 'label'
})
alphafold_disorder_benchmark_sub.to_csv(data_path + "disorder_metrics.csv", index = False)