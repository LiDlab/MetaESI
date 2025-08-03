import pandas as pd
from functools import reduce
from combine_EMD_result import rename_and_select_columns

# Define data path and load data from different prediction methods
data_path = "../../results/interface_benchmark/"
MetaESI = pd.read_csv(data_path + "Interface_EMD_by_MetaESI.csv")
MetaESIwoSeq = pd.read_csv(data_path + "Interface_EMD_by_MetaESI_woSeq.csv")
MetaESIwo3D = pd.read_csv(data_path + "Interface_EMD_by_MetaESI_wo3D.csv")
MetaESIwoGARD = pd.read_csv(data_path + "Interface_EMD_by_MetaESI_woGARD.csv")
MetaESIwoMeta = pd.read_csv(data_path + "Interface_EMD_by_MetaESI_woMeta.csv")

# Process MetaESI data with direct column selection
MetaESI = MetaESI[['ELMIdentifier', 'E3_Entry', 'Site', 'sub_Entry', 'Start', 'End', 'MetaESI Interface Residues (E3)', 'MetaESI Interface Residues (SUB)',
                   'MetaESI EMD 2D', 'MetaESI EMD E3', 'MetaESI EMD SUB', 'Random EMD 2D', 'Random EMD E3', 'Random EMD SUB',]].copy()

# Process other datasets
MetaESIwoSeq = rename_and_select_columns(MetaESIwoSeq, 'MetaESIwoSeq')
MetaESIwo3D = rename_and_select_columns(MetaESIwo3D, 'MetaESIwo3D')
MetaESIwoGARD = rename_and_select_columns(MetaESIwoGARD, 'MetaESIwoGARD')
MetaESIwoMeta = rename_and_select_columns(MetaESIwoMeta, 'MetaESIwoMeta')

# Combine multiple DataFrames using merge operations
dfs = [MetaESI, MetaESIwoSeq, MetaESIwo3D, MetaESIwoGARD, MetaESIwoMeta]
combined_df = reduce(lambda left, right: pd.merge(left, right, on=['ELMIdentifier', 'E3_Entry', 'Site', 'sub_Entry', 'Start', 'End'], how='outer'), dfs)
combined_df['Start'] = combined_df['Start'].astype(str) + '-' + combined_df['End'].astype(str)
combined_df.drop(columns=['End'], inplace=True)

# Rename final columns
combined_df = combined_df.rename(columns={'E3_Entry': 'SwissProt ID (E3)', 'Site': 'Interface Residues (E3)', 'sub_Entry': 'SwissProt ID (SUB)',
                        'Start': 'Interface Residue (SUB)'})

# Save combined DataFrame to CSV
combined_df.to_csv(data_path + "Interface_EMD_combined_ablation.csv", index = False)
