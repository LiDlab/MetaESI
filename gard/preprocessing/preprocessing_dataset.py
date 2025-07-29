import os
import glob
import pandas as pd
import numpy as np

"""
Processes protein disorder annotations and creates benchmark datasets.

Key Functions:
- txt2list/list2txt: Helper functions for reading/writing text files
- extract_region_boundaries: Parses UniProt boundary strings into start/end positions
- format_region_boundaries: Creates position-level annotation DataFrames
- get_disorder_annotation: Generates binary annotations for disordered/ordered regions

Workflow:
1. Loads AlphaFold2 processed proteins (from HDF files)
2. Processes two data sources:
   - DisProt database (disordered regions)
   - Ordered structures dataset
3. Filters proteins to only those with AlphaFold2 predictions
4. Creates position-level annotations (1=disordered, 0=ordered)
5. Merges and cleans annotations (resolving conflicts)
6. Saves:
   - Ground truth annotations (CSV)
   - Benchmark protein list (TXT)

Output Files:
- groundtruth_data.csv: Position-level disorder annotations
- benchmark_proteins.txt: List of evaluated proteins
"""

def txt2list(name):
    benchmark = open(name+".txt", "r")
    ls = benchmark.readlines()
    ls = [x.strip() for x in ls]
    benchmark.close()

    return ls

def list2txt(ls, name):
    benchmark = open(name+".txt", "w")
    for protein in ls:
        benchmark.write(protein + "\n")
    benchmark.close()

def extract_region_boundaries(df):
    df["start"] = df["UniProt boundaries"].apply(lambda x: int(x.split('-')[0]))
    df["end"] = df["UniProt boundaries"].apply(lambda x: int(x.split('-')[1]))
    df = df[["UniProt accession", "start", "end"]]
    df = df.rename(columns={"UniProt accession": "protein_id"})
    return df

def format_region_boundaries(prot, start, end, column_name):
    try:
        position = np.arange(start,end)
        res = pd.DataFrame({"protein_id": np.repeat(prot, len(position)),
                            "position": position,
                            column_name: np.repeat(1, len(position))})
    except:
        res = None
    return(res)

def get_disorder_annotation(df, column_name):
    res = df.apply(lambda row : format_region_boundaries(row['protein_id'],row['start'], row['end'], column_name), axis = 1)
    res_filtered = [r for r in res if r is not None]
    return pd.concat(res_filtered)




data_path = "../../data/human/raw/disordered/"
save_path = "../../results/idr_benchmark/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

af2_path = "../../data/human/raw/alphafold2/"
af2_files = glob.glob(af2_path + '*.hdf')
all_entry = [i.split('_')[-1].split('.')[0] for i in af2_files]

disordered_data = pd.read_csv(data_path + "DisProt_release_2022_06.tsv",sep="\t")
disordered_data = disordered_data[["acc", "start", "end"]]
disordered_data = disordered_data.rename(columns={"acc": "protein_id"})
disordered_data = disordered_data.drop_duplicates()
# filter
disordered_data = disordered_data[disordered_data["protein_id"].isin(all_entry)]    # 去掉同源异构体

disordered_data_annotation = get_disorder_annotation(df=disordered_data,column_name="disordered")
disordered_data_annotation = disordered_data_annotation.drop_duplicates()

ordered_data = pd.read_csv(data_path + 'ordered_structures.csv',sep=";")
ordered_data = extract_region_boundaries(ordered_data)
# filter
ordered_data = ordered_data[ordered_data["protein_id"].isin(all_entry)]

ordered_data_annotation = get_disorder_annotation(df=ordered_data,column_name="ordered")

groundtruth_data = disordered_data_annotation.merge(ordered_data_annotation, how='outer', on = ['protein_id','position'])
groundtruth_data = groundtruth_data.fillna(0)
groundtruth_data = groundtruth_data.drop_duplicates()

groundtruth_data = groundtruth_data[groundtruth_data.disordered!=groundtruth_data.ordered].reset_index(drop=True)

benchmark_proteins = groundtruth_data.protein_id.unique()

groundtruth_data.to_csv(save_path + 'groundtruth_data.csv', index=False)
list2txt(benchmark_proteins, save_path + "benchmark_proteins")