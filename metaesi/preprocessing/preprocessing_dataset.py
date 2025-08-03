import pandas as pd
from tqdm import tqdm
from random import sample
import h5py

"""
Constructs balanced dataset for E3-substrate interaction (ESI) prediction.

Key Features:
1. Data Integration:
   - Combines positive ESI interactions from Ubibrowser2 and UbiNet databases
   - Uses STRING protein interactions (score >300) for negative sampling
   - Maps protein identifiers to UniProt accessions

2. Sampling Strategy:
   - Performs two-stage negative sampling:
     * E3-balanced: Matches #negatives to #positives per E3 enzyme
     * Substrate-balanced: Matches #negatives to #positives per substrate
   - Prioritizes STRING interactors as negatives before random sampling

3. Data Processing:
   - Standardizes publication references
   - Removes duplicate interactions
   - Filters proteins with available structural features
   - Generates unique interaction keys

Output Files:
- esi_count_per_e3.csv: Substrate counts per E3
- dataset.csv: Final balanced dataset containing:
  * Positive interactions (GSP)
  * E3-balanced negatives (GSN1)
  * Substrate-balanced negatives (GSN2)
  * Interaction metadata and keys

Workflow:
1. Loads and merges positive ESI datasets
2. Processes STRING interactions for negatives
3. Performs tiered negative sampling
4. Generates balanced dataset
5. Saves final CSV file
"""

data_path = "../../data/human/raw/esi/"
feature_path = "../../data/human/features/"
save_path = "../../data/human/dataset/"

e3_features = h5py.File(feature_path + "MetaESI_features_e3.hdf5", "r")
all_e3_entry = list(e3_features.keys())
e3_features.close()
sub_features = h5py.File(feature_path + "MetaESI_features_sub.hdf5", "r")
all_sub_entry = list(sub_features.keys())
sub_features.close()
uniprot = pd.read_table(data_path + "uniprot_20230511.tsv")


############ GSP
ubibrowser2 = pd.read_table(data_path + "literature.E3.txt")
ubibrowser2_human = ubibrowser2[ubibrowser2["species"] == "H.sapiens"].copy()
ubibrowser2_human[["E3","SUB","source"]] = ubibrowser2_human[["SwissProt AC (E3)","SwissProt AC (Substrate)","SOURCEID"]]
ubibrowser2_human = ubibrowser2_human[["E3","SUB","source"]].copy()

ubinet2_human = pd.read_excel(data_path+"UbiNet_2.xlsx", sheet_name="Human")
ubinet2_human[["E3","SUB","source"]] = ubinet2_human[["E3_AC","SUB_AC","PMID"]]
ubinet2_human = ubinet2_human[["E3","SUB","source"]].copy()

ubibrowser2_human = ubibrowser2_human[(ubibrowser2_human["E3"] != "-")&(ubibrowser2_human["SUB"] != "-")].copy()
ubinet2_human = ubinet2_human[(ubinet2_human["E3"] != "-")&(ubinet2_human["SUB"] != "-")].copy()

esi = pd.concat([ubinet2_human,ubibrowser2_human])
esi.reset_index(inplace = True,drop = True)

for i in range(len(esi)):
    if "_HUMAN" in esi.loc[i]["source"]:
        esi.loc[i]["source"] = 11111111
    elif ":" in esi.loc[i]["source"]:
        esi.loc[i]["source"] = int(min(esi.loc[i]["source"].split(":")))
    elif ";" in esi.loc[i]["source"]:
        esi.loc[i]["source"] = int(min(esi.loc[i]["source"].split(";")))
    elif "PMC4306233" in esi.loc[i]["source"]:
        esi.loc[i]["source"] = 24658274
    else:
        esi.loc[i]["source"] = int(esi.loc[i]["source"])

esi.sort_values(by="source", inplace = True)
esi.drop_duplicates(subset=["E3","SUB"],keep="first", inplace = True, ignore_index=True)

esi = esi[(esi["E3"].isin(all_e3_entry))&(esi["SUB"].isin(all_sub_entry))]
esi.reset_index(inplace = True,drop = True)


############ ESI_count per E3
esi_count = pd.DataFrame({'E3': esi['E3'].value_counts().index, 'count': esi['E3'].value_counts().values})
esi_count.to_csv(save_path + "esi_count_per_e3.csv", index = False)


############ GSN
string = pd.read_table(data_path + "9606.protein.links.v11.5.txt", delimiter=" ")
string = string[string['combined_score'] > 300]

string2entry = uniprot[['Entry', 'STRING']].copy()
string2entry.dropna(axis = 0, how='any', inplace = True)
string2entry['STRING'] = string2entry['STRING'].apply(lambda x: str(x)[:-1])

string = string[(string['protein1'].isin(string2entry['STRING'].values))&(string['protein2'].isin(string2entry['STRING'].values))]

entry_mapping = dict(zip(list(string2entry['STRING'].values),
                     list(string2entry['Entry'].values)))
string.loc[:, 'protein1'] = string['protein1'].apply(lambda x:entry_mapping[x])
string.loc[:, 'protein2'] = string['protein2'].apply(lambda x:entry_mapping[x])

string[['P1','P2']] = string[['protein1','protein2']]
string = string[['P1','P2']].copy()
string_copy = string[['P1','P2']].copy()
string_copy.rename(columns = dict(zip(['P2', 'P1'], ['P1', 'P2'])), inplace = True)
string = pd.concat([string, string_copy], axis = 0)
string.drop_duplicates(keep="first", inplace=True, ignore_index=True)

string = string[(string['P1'].isin(all_e3_entry))&(string['P2'].isin(all_sub_entry))].copy()

gsp = []
gsn1 = []
gsn2 = []

############ sample GSN by E3
all_e3 = list(esi_count["E3"].values)
for i in tqdm(all_e3):
    ith_e3_esi = esi[esi['E3'] == i]
    ith_e3_sub = set(ith_e3_esi['SUB'])
    if len(ith_e3_sub) == 0:
        continue

    ith_e3_pro = set(string.loc[string['P1'] == i, 'P2'])
    ith_e3_pro = ith_e3_pro - ith_e3_sub
    ith_e3_pro = list(ith_e3_pro)

    ith_e3_pro_all = []

    add_num = len(ith_e3_sub)
    if len(ith_e3_pro) >= add_num:
        sample_pro = sample(ith_e3_pro, add_num)
        ith_e3_pro_all += sample_pro

    else:
        if len(ith_e3_pro) != 0:
            ith_e3_pro_all += ith_e3_pro

        add_num = add_num - len(ith_e3_pro)

        # sample random protein
        ith_e3_random = list(set(all_sub_entry) - ith_e3_sub - set(ith_e3_pro) - set(all_e3))
        add_pro = sample(ith_e3_random, add_num)
        ith_e3_pro_all += add_pro

    ith_e3_epi = {"E3":[i]*len(ith_e3_pro_all),"SUB_random":ith_e3_pro_all}
    ith_e3_epi = pd.DataFrame(ith_e3_epi)

    ith_e3_esi = ith_e3_esi.assign(key=ith_e3_esi['E3'] + '_' + ith_e3_esi['SUB'])
    ith_e3_esi.reset_index(drop = True, inplace = True)
    ith_e3_epi = pd.concat([ith_e3_epi, ith_e3_esi[['key']]], axis = 1)

    gsp.append(ith_e3_esi)
    gsn1.append(ith_e3_epi)

gsp = pd.concat(gsp, ignore_index = True)
gsn1 = pd.concat(gsn1, ignore_index = True)


############ sample GSN by SUB
all_sub = list(set(esi["SUB"].values))
for i in tqdm(all_sub):
    ith_sub_esi = esi[esi['SUB'] == i]
    ith_sub_e3 = set(ith_sub_esi['E3'])
    if len(ith_sub_e3) == 0:
        continue

    ith_sub_gsn1 = set(gsn1.loc[gsn1['SUB_random'] == i, 'E3'])

    ith_sub_pro = set(string.loc[string['P2'] == i, 'P1'])
    ith_sub_pro = ith_sub_pro - ith_sub_e3 - ith_sub_gsn1
    ith_sub_pro = list(ith_sub_pro)

    ith_sub_pro_all = []

    add_num = len(ith_sub_e3)
    if len(ith_sub_pro) >= add_num:
        sample_pro = sample(ith_sub_pro, add_num)
        ith_sub_pro_all += sample_pro

    else:
        if len(ith_sub_pro) != 0:
            ith_sub_pro_all += ith_sub_pro

        add_num = add_num - len(ith_sub_pro)

        ith_sub_random = list(set(all_e3_entry) - ith_sub_e3 - set(ith_sub_pro))
        add_pro = sample(ith_sub_random, add_num)
        ith_sub_pro_all += add_pro

    ith_sub_psi = {"E3_random":ith_sub_pro_all, "SUB":[i]*len(ith_sub_pro_all)}
    ith_sub_psi = pd.DataFrame(ith_sub_psi)

    ith_sub_esi = ith_sub_esi.assign(key=ith_sub_esi['E3'] + '_' + ith_sub_esi['SUB'])

    ith_sub_esi.reset_index(drop = True, inplace = True)
    ith_sub_psi = pd.concat([ith_sub_psi, ith_sub_esi[['key']]], axis = 1)

    gsn2.append(ith_sub_psi)

gsn2 = pd.concat(gsn2, ignore_index = True)

df = pd.merge(gsp, gsn1, on = ['E3', 'key'], how = 'inner')
df = pd.merge(df, gsn2, on = ['SUB', 'key'], how = 'inner')

df.to_csv(save_path + "dataset.csv", index = False, header = True)