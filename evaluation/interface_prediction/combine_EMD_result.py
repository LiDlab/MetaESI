import pandas as pd
from functools import reduce

def rename_and_select_columns(df, new_name):
    """
    Add columns containing 'new_name' and select specific columns from DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame to process
        new_name (str): New identifier to add 'new_name' in column names

    Returns:
        pd.DataFrame: Processed DataFrame with renamed and selected columns
    """
    new_columns = {
        'Interface Residues (E3)': f'{new_name} Interface Residues (E3)',
        'Interface Residues (SUB)': f'{new_name} Interface Residues (SUB)',
        'EMD 2D': f'{new_name} EMD 2D',
        'EMD E3': f'{new_name} EMD E3',
        'EMD SUB': f'{new_name} EMD SUB'
    }

    df = df.rename(columns=new_columns)

    # Select columns with specific order
    df = df[[
        'ELMIdentifier', 'E3_Entry', 'Site', 'sub_Entry', 'Start', 'End',
        f'{new_name} Interface Residues (E3)',
        f'{new_name} Interface Residues (SUB)',
        f'{new_name} EMD 2D',
        f'{new_name} EMD E3',
        f'{new_name} EMD SUB'
    ]].copy()

    return df

# Define data path and load data from different prediction methods
data_path = "../../results/interface_benchmark/"
MetaESI = pd.read_csv(data_path + "Interface_EMD_by_MetaESI.csv")
AlphaFold = pd.read_csv(data_path + "Interface_EMD_by_AF3.csv")
ESMFold = pd.read_csv(data_path + "Interface_EMD_by_ESM.csv")
ZDOCK = pd.read_csv(data_path + "Interface_EMD_by_ZDOCK.csv")
PeSTo = pd.read_csv(data_path + "Interface_EMD_by_PeSTo.csv")
ScanNet = pd.read_csv(data_path + "Interface_EMD_by_ScanNet.csv")

# Process MetaESI data with direct column selection
MetaESI = MetaESI[['ELMIdentifier', 'E3_Entry', 'Site', 'sub_Entry', 'Start', 'End', 'MetaESI Interface Residues (E3)', 'MetaESI Interface Residues (SUB)',
                   'MetaESI EMD 2D', 'MetaESI EMD E3', 'MetaESI EMD SUB', 'Random EMD 2D', 'Random EMD E3', 'Random EMD SUB',]].copy()

# Process other datasets
AlphaFold = rename_and_select_columns(AlphaFold, 'AlphaFold')
ESMFold = rename_and_select_columns(ESMFold, 'ESMFold')
ZDOCK = rename_and_select_columns(ZDOCK, 'ZDOCK')
PeSTo = rename_and_select_columns(PeSTo, 'PeSTo')
ScanNet = rename_and_select_columns(ScanNet, 'ScanNet')

# Combine multiple DataFrames using merge operations
dfs = [MetaESI, AlphaFold, ESMFold, ZDOCK, PeSTo, ScanNet]
combined_df = reduce(lambda left, right: pd.merge(left, right, on=['ELMIdentifier', 'E3_Entry', 'Site', 'sub_Entry', 'Start', 'End'], how='outer'), dfs)
combined_df['Start'] = combined_df['Start'].astype(str) + '-' + combined_df['End'].astype(str)
combined_df.drop(columns=['End'], inplace=True)

# Rename final columns
combined_df = combined_df.rename(columns={'E3_Entry': 'SwissProt ID (E3)', 'Site': 'Interface Residues (E3)', 'sub_Entry': 'SwissProt ID (SUB)',
                        'Start': 'Interface Residue (SUB)'})

# Save combined DataFrame to CSV
combined_df.to_csv(data_path + "Interface_EMD_combined.csv", index = False)
