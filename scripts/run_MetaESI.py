import argparse
import pathlib
import subprocess
import sys
import os
import glob
import io
from contextlib import redirect_stdout
from Bio import PDB

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from gard.preprocessing.preprocessing_features import cal_RD, cal_adj
from metaesi.preprocessing.preprocessing_features import extract_metaesi_feature
from metaesi import *
import json
import numpy as np
import h5py
import pandas as pd
import torch

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def create_parser():
    """
    Create and configure the argument parser for command-line options.

    Parameters:
        None

    Returns:
        argparse.ArgumentParser: Configured argument parser object
    """

    parser = argparse.ArgumentParser(
        description="Predict interaction probability and interface residues between E3 ligase and substrate proteins using MetaESI"
    )

    parser.add_argument(
        "-e", "--e3",
        type=str,
        required = True,
        help="[REQUIRED] UniProt ID of the E3 ubiquitin ligase (e.g., Q8CFI0)",
    )
    parser.add_argument(
        "-s", "--candidate_sub",
        type=str,
        required=True,
        help="[REQUIRED] UniProt ID of candidate substrate protein (e.g., P35583)",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=pathlib.Path,
        required=True,
        help="[REQUIRED] Output directory path for saving: "
         "1) Prediction results "
         "2) Visualizations "
         "3) Annotated PDB files",
    )
    parser.add_argument(
        "-g", "--gpu",
        type=int,
        default=0,
        help="Specify GPU device ID for acceleration (default: 0). "
             "Use -1 to force CPU-only execution"
    )

    return parser


def run_download_script(pro_id, output_dir):
    """
    Download AlphaFold structure files for a protein using a bash script.

    Parameters:
        pro_id (str): UniProt ID of the protein to download
        output_dir (pathlib.Path): Directory to save downloaded files

    Returns:
        None
    """

    script_path = pathlib.Path(__file__).parent / "download_af_pae_per_pro.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"Bash script not found at {script_path}")

    cmd = [
        "/bin/bash",
        str(script_path),
        pro_id,
        str(output_dir.resolve())
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"  • PDB: {output_dir}/AF-{pro_id}-F1-model_v4.pdb")
        print(f"  • PAE: {output_dir}/AF-{pro_id}-F1-predicted_aligned_error_v4.json")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Download failed: {e}")
        sys.exit(1)


def run_esm2_script(output_dir, fasta_file_name):
    """
    Generate ESM-2 features for protein sequences using a bash script.

    Parameters:
        output_dir (pathlib.Path): Directory containing input FASTA and output location
        fasta_file_name (str): Name of the FASTA file to process

    Returns:
        None
    """

    script_path = pathlib.Path(__file__).parent / "download_esm2_per_pro.sh"

    if not script_path.exists():
        raise FileNotFoundError(f"Bash script not found at {script_path}")

    cmd = [
        "/bin/bash",
        str(script_path),
        str(output_dir.resolve()),
        fasta_file_name
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Feature generation failed: {e}")
        sys.exit(1)


def extract_sequence_from_pdb(pdb_file):
    """
    Extract amino acid sequence from a PDB structure file.

    Parameters:
        pdb_file (str): Path to PDB structure file

    Returns:
        str: Protein sequence as single-letter codes
    """

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    sequence = ''

    for model in structure:
        for chain in model:
            polypeptides = PDB.PPBuilder().build_peptides(chain)
            for poly in polypeptides:
                sequence += str(poly.get_sequence())

    return sequence


def load_error_distance(AC, data_path):
    """
    Load predicted aligned error (PAE) from AlphaFold JSON output.

    Parameters:
        AC (str): UniProt ID of the protein
        data_path (pathlib.Path): Directory containing PAE JSON file

    Returns:
        numpy.ndarray: 2D array of predicted aligned error distances
    """

    with open(data_path / "AF-{}-F1-predicted_aligned_error_v4.json".format(AC)) as f_pae:
        data = json.load(f_pae)
    error_dist = np.array(data[0]["predicted_aligned_error"])

    return error_dist


def load_features(AC, data_path):
    """
    Calculate structural features (residue depth and adjacency matrix) from PDB.

    Parameters:
        AC (str): UniProt ID of the protein
        data_path (pathlib.Path): Directory containing PDB structure file

    Returns:
        tuple: Tuple containing (protein ID, residue depth, adjacency matrix, PAE)
    """

    pdbfile = data_path / "AF-{}-F1-model_v4.pdb".format(AC)

    parser = PDB.PDBParser()
    structure = parser.get_structure(AC, pdbfile)
    model = structure[0]

    RD = cal_RD(model)
    A = cal_adj(structure)
    PAE = load_error_distance(AC, data_path)

    if RD.shape[0] != A.shape[0]:
        print("ERROR:" + AC)

    return (AC,RD,A,PAE)


def fetch_e3_specific_model(e3, gpu,
                            meta_model_path,
                            e3_model_path,
                            meta_model_name = "PanESI.pth"):
    """
    Load E3-specific model if unavailable.

    Parameters:
        e3 (str): UniProt ID of the E3 ligase
        gpu (int): GPU device index
        meta_model_path (str): Path to meta-model weights
        e3_model_path (str): Directory for E3-specific models
        meta_model_name (str): Filename of meta-model weights

    Returns:
        MetaESI: Initialized model loaded on appropriate device
    """

    e3_model = os.listdir(e3_model_path)

    if e3 + '.pth' in e3_model:
        # Load the existing E3-specific learner
        model = MetaESI(1280).to(device=try_gpu(gpu))
        model_dict = torch.load(e3_model_path + e3 + '.pth', map_location=try_gpu(gpu))
        model.load_state_dict(model_dict)

    else:
        # Fine-tune and save the E3-specific learner
        model = MetaESI(1280).to(device=try_gpu(gpu))
        model_dict = torch.load(meta_model_path + meta_model_name, map_location=try_gpu(gpu))
        model.load_state_dict(model_dict)

    return model


def merge_residues(residues):
    """
    Merge consecutive residue numbers into ranges (e.g., 1,2,3,5 → "1-3;5").

    Parameters:
        residues (array-like): Array of residue positions

    Returns:
        str: Merged residue range string
    """

    if len(residues) == 0:
        return ""

    residues = np.sort(residues)

    merged = []
    start = residues[0]
    prev = residues[0]

    for i in range(1, len(residues)):
        if residues[i] - prev <= 5:
            prev = residues[i]
        else:
            if start == prev:
                merged.append(f"{int(start)}")
            else:
                merged.append(f"{int(start)}-{int(prev)}")
            start = residues[i]
            prev = residues[i]

    if start == prev:
        merged.append(f"{int(start)}")
    else:
        merged.append(f"{int(start)}-{int(prev)}")

    return "; ".join(merged)


def map_residues_to_sequence(sequence, residue_ranges):
    """
    Map residue ranges to their corresponding sequence positions and amino acids.

    Parameters:
        sequence (str): Protein sequence
        residue_ranges (str): Semi-colon separated residue range string

    Returns:
        str: Formatted residue positions with amino acids (e.g., S56;L57-59)
    """

    ranges = residue_ranges.split("; ")
    result = []

    for r in ranges:
        if '-' in r:
            start, end = map(int, r.split('-'))
            if start > len(sequence) or end > len(sequence):
                continue
            start_residue = sequence[start - 1]
            end_residue = sequence[end - 1]
            result.append(f"{start_residue}{start}-{end_residue}{end}")
        else:
            pos = int(r)
            if pos > len(sequence):
                continue
            residue = sequence[pos - 1]
            result.append(f"{residue}{pos}")

    return "; ".join(result)



def get_crop_boundaries(i_map):
    """
    Identify boundaries for cropping a 21x21 hotspot region centered at maximum value.

    Parameters:
        i_map (numpy.ndarray): 2D interaction probability map

    Returns:
        tuple: Boundary coordinates (top, bottom, left, right) for cropping
    """

    H, W = i_map.shape
    row, col = np.unravel_index(np.argmax(i_map), i_map.shape)

    top = max(0, row - 10)
    bottom = top + 21

    if bottom > H:
        bottom = H
        top = H - 21

    left = max(0, col - 10)
    right = left + 21

    if right > W:
        right = W
        left = W - 21

    return top, bottom, left, right


def update_pdb_bfactor(pdb_file, protein_scores, output_file):
    """
    Update B-factor column in PDB file with per-residue interface scores.

    Parameters:
        pdb_file (str): Path to input PDB file
        protein_scores (dict): Residue-position to score mapping
        output_file (pathlib.Path): Path to modified PDB output file

    Returns:
        None
    """

    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    updated_lines = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):

            res_seq = int(line[22:26].strip())
            score = protein_scores.get(res_seq, -1)

            updated_line = line[:60] + f"{score:6.2f}" + line[66:]
            updated_lines.append(updated_line)
        else:
            updated_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(updated_lines)


def run(args):
    """
    Execute full prediction pipeline: data download, feature extraction, prediction, and output generation.

    Parameters:
        args (Namespace): Command-line arguments parsed by argparse

    Returns:
        None
    """

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("         MetaESI Prediction Pipeline")
    print("=" * 50)

    # Download AlphaFold structures for both proteins
    print(f"\n[1/4] Downloading Protein Structures")
    print("-" * 50)
    print(f"✓ Downloaded E3 Structure ({args.e3})")
    run_download_script(args.e3, args.output_dir)
    print(f"✓ Downloaded Substrate Structure ({args.candidate_sub})")
    run_download_script(args.candidate_sub, args.output_dir)

    protein_sequences = {}

    # Process E3 protein
    print(f"\n[2/4] Processing Protein Sequence")
    print("-" * 50)
    e3_pdb = args.output_dir / f"AF-{args.e3}-F1-model_v4.pdb"
    if e3_pdb.exists():
        e3_seq = extract_sequence_from_pdb(str(e3_pdb))
        protein_sequences[args.e3] = e3_seq
    else:
        print(f"✗ Error: E3 PDB file not found: {e3_pdb}")
        sys.exit(1)

    # Process substrate protein
    sub_pdb = args.output_dir / f"AF-{args.candidate_sub}-F1-model_v4.pdb"
    if sub_pdb.exists():
        sub_seq = extract_sequence_from_pdb(str(sub_pdb))
        protein_sequences[args.candidate_sub] = sub_seq

    else:
        print(f"✗ Error: Substrate PDB file not found: {sub_pdb}")
        sys.exit(1)

    # Generate combined FASTA file
    fasta_file = args.output_dir / f"{args.e3}_{args.candidate_sub}.fasta"
    with open(fasta_file, 'w') as f:
        for prot_id, seq in protein_sequences.items():
            f.write(f">{prot_id}\n{seq}\n")
    print(f"✓ Generated FASTA file: {fasta_file.name}")
    print(f"  • E3 ({args.e3}): {len(e3_seq)} residues")
    print(f"  • Substrate ({args.candidate_sub}): {len(sub_seq)} residues")

    # Generate ESM2 features using FASTA
    print(f"\n[3/4] Extracting Protein Features")
    print("-" * 50)
    run_esm2_script(args.output_dir, f"{args.e3}_{args.candidate_sub}.fasta")
    print(f"✓ Extracted ESM-2 features")

    f = h5py.File(args.output_dir / "GARD_features.hdf5", "a") # Store GARD structural features in HDF5
    for protein in [args.e3, args.candidate_sub]:
        if protein in f:
            continue
        features = load_features(protein, args.output_dir)
        group = f.create_group(features[0])
        group.create_dataset("RD", data=features[1])
        group.create_dataset("A", data=features[2])
        group.create_dataset("PAE", data=features[3])
    f.close()
    print(f"✓ Extracted GARD features")

    # Extract MetaESI features from GARD inputs
    with io.StringIO() as buf, redirect_stdout(buf):
        extract_metaesi_feature(args.output_dir, args.output_dir, fasta_file, verbose=False)
    print("✓ Extracted MetaESI features")

    # Initialize prediction pipeline
    print(f"\n[4/4] Running Prediction Model")
    print("-" * 50)
    gpu = args.gpu
    e3_features = h5py.File(os.path.join(args.output_dir, "METAESI_features_e3.hdf5"), "r")
    sub_features = h5py.File(os.path.join(args.output_dir, "METAESI_features_sub.hdf5"), "r")
    Feature = ESIfeature(e3_features, sub_features, c_map_thr=9)
    dataset_query = np.array([[args.e3, args.candidate_sub, 1]])
    Feature.load_ESIs(dataset_query, shuffle=False)

    # Load E3-specific prediction model
    model = fetch_e3_specific_model(args.e3, gpu = gpu, meta_model_path = os.path.join(project_root, "models/meta_model/"), e3_model_path = os.path.join(project_root, "models/e3_specific_model/"))
    print(
        f"✓ Loaded {'E3-specific' if args.e3 + '.pth' in os.listdir(os.path.join(project_root, 'models/e3_specific_model/')) else 'Meta'} model for {args.e3}")

    # Process each feature set
    for idx, (e3_E, e3_A, sub_E_list, label) in enumerate(Feature):
        # Skip if no disordered regions found in substrate
        if len(sub_E_list) == 0:
            print("⚠ Skipping prediction: No disordered regions found in substrate")
            continue

        # Generate interface map prediction
        model.eval()
        i_map = []
        for sub_idx in range(len(sub_E_list)):
            sub_E = sub_E_list[sub_idx]
            with torch.no_grad():
                cm = model(e3_E.to(device=try_gpu(gpu)), e3_A.to(device=try_gpu(gpu)), sub_E.to(device=try_gpu(gpu)))
            i_map.append(cm)
        print(f"✓ Calculated {args.e3}-{args.candidate_sub} MetaESI interface map")

        # Process prediction results
        i_map = torch.cat(i_map, 1)
        p_hat = torch.max(i_map)
        i_map = i_map.cpu().squeeze().data.numpy()
        order1, order2 = Feature.get_order(dataset_query, idx)
        seq1, seq2 = Feature.get_seq(dataset_query, idx)
        pred1, pred2 = extract_interface_residues(i_map, p_hat, order1, order2, threshold=0.1)
        p_hat = p_hat.item()

        # Calculate and normalize prediction score
        logits = np.log(p_hat / (1 - p_hat))
        scaled_scores = 1 / (1 + np.exp(-0.5 * logits))
        scaled_scores = min(0.999, scaled_scores.round(3))
        print(f"✓ Calculated {args.e3}-{args.candidate_sub} MetaESI score")

        # Check MetaESI-Atlas for existing predictions
        metaeai_atlas_path = os.path.join(project_root, "results", "metaesi_atlas")
        csv_files = glob.glob(os.path.join(metaeai_atlas_path, "*.csv"))
        dfs = []
        for file_path in csv_files:
            df = pd.read_csv(file_path)
            dfs.append(df)
        metaesi_atlas = pd.concat(dfs, axis=0)

        results = metaesi_atlas[(metaesi_atlas['SwissProt ID (E3)'] == args.e3) &
                                (metaesi_atlas['SwissProt ID (SUB)'] == args.candidate_sub)]

        # Use Atlas data if available
        if not results.empty:
            scaled_scores = results['MetaESI Score'].values[0]
            s = np.clip(scaled_scores, 1e-7, 1 - 1e-7)
            numerator = s ** 2
            denominator = 2 * s ** 2 - 2 * s + 1
            p_hat = numerator / denominator

        # Assign confidence level based on raw score
        if p_hat < 0.5:
            confidence = "very low"
        elif 0.5 <= p_hat < 0.74968:
            confidence = "low"
        elif 0.74968 <= p_hat < 0.85:
            confidence = "Medium"
        elif 0.85 <= p_hat < 0.97:
            confidence = "high"
        else:
            confidence = "very high"





        # Print prediction results
        print("\n\n" + "=" * 50)
        print("               PREDICTION RESULTS")
        print("=" * 50)

        RED = "\033[91m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        RESET = "\033[0m"

        print(f"{RED}🔬 MetaESI Score{RESET}")
        print(f"  • {scaled_scores} ({confidence} confidence)")

        # Display interface residues for medium+ confidence predictions
        if p_hat >= 0.74968:
            if not results.empty:
                e3_res = results['Interface Residues (E3)'].values[0]
                sub_res = results['Interface Residues (SUB)'].values[0]
            else:
                e3_imp = np.array(pred1)
                sub_imp = np.array(pred2)
                e3_region = merge_residues(e3_imp)
                sub_region = merge_residues(sub_imp)
                e3_res = map_residues_to_sequence(e3_seq, e3_region)
                sub_res = map_residues_to_sequence(sub_seq, sub_region)

            print(f"\n{RED}🔥 Interface Residues{RESET}")
            print(f"  • E3 ({args.e3}): {e3_res}")
            print(f"  • Substrate ({args.candidate_sub}): {sub_res}")

        #############################################################################
        # Generate full interface map visualization
        print(f"\n{GREEN}📁 Output Files Generated{RESET}")

        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        colors = ['#D2ECF4', '#FEFEBE', '#FED283', '#F88C51', '#DD3E2D', '#A50026']
        custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

        plt.figure(figsize=(16, 16))
        font = 32

        i_map_reverse = i_map[::-1, :]
        ax = sns.heatmap(i_map_reverse, cmap=custom_cmap, vmin=0, vmax=1, xticklabels=False, yticklabels=False, cbar=False)

        # Configure plot aesthetics
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1)  # 设置边框线宽
        ax.spines['right'].set_linewidth(1)  # 设置边框线宽
        ax.spines['top'].set_linewidth(1)  # 设置边框线宽
        ax.spines['bottom'].set_linewidth(1)  # 设置边框线宽
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')

        plt.axis('scaled')
        plt.title(f"{args.e3}_{args.candidate_sub} MetaESI interface map", fontsize=font)
        plt.ylabel(args.e3, fontsize=font)
        plt.xlabel(args.candidate_sub, fontsize=font)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{args.e3}_{args.candidate_sub}_imap.pdf"), format='pdf')
        plt.savefig(os.path.join(args.output_dir, f"{args.e3}_{args.candidate_sub}_imap.png"), format='png')

        print(f"  • Full interface map: {args.e3}_{args.candidate_sub}_imap.pdf")

        #############################################################################
        # Generate zoomed interface hotspot visualization
        top, bottom, left, right = get_crop_boundaries(i_map)
        e3_start = int(order1[top])
        e3_end = int(order1[bottom])
        sub_start = int(order2[left])
        sub_end = int(order2[right])

        hotspot = i_map[top:bottom + 1, left:right + 1]
        seq1_hot = seq1[e3_start - 1:e3_end]
        seq1_hot = [f'{char}{e3_start + i}' for i, char in enumerate(seq1_hot)]
        seq1_hot = seq1_hot[::-1]
        seq2_hot = seq2[sub_start - 1:sub_end]
        seq2_hot = [f'{char}{sub_start + i}' for i, char in enumerate(seq2_hot)]

        plt.figure(figsize=(8, 8))
        font = 16

        hotspot = hotspot[::-1, :]
        ax = sns.heatmap(hotspot, cmap=custom_cmap, vmin=0, vmax=1, xticklabels=seq2_hot, yticklabels=seq1_hot, linewidths=.5,
                         cbar=False)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=font - 4)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=font - 4)

        ax.tick_params(axis='x', labelsize=font - 4, labelrotation=90)

        # Configure plot aesthetics
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['top'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.spines['left'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['bottom'].set_color('black')

        plt.axis('scaled')
        plt.title(f"{args.e3}_{args.candidate_sub} MetaESI interface hotspot", fontsize=font)
        plt.ylabel(args.e3, fontsize=font)
        plt.xlabel(args.candidate_sub, fontsize=font)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"{args.e3}_{args.candidate_sub}_hotspot.pdf"), format='pdf')
        plt.savefig(os.path.join(args.output_dir, f"{args.e3}_{args.candidate_sub}_hotspot.png"), format='png')

        print(f"  • Interface hotspot: {args.e3}_{args.candidate_sub}_hotspot.pdf")

        #############################################################################
        # Prepare PDB files with interface scores for visualization

        protein1_scores = {}
        for i, row in enumerate(i_map):
            max_score = np.max(row)
            protein1_scores[int(order1[i])] = max_score

        protein2_scores = {}
        for j, col in enumerate(i_map.T):
            max_score = np.max(col)
            protein2_scores[int(order2[j])] = max_score

        update_pdb_bfactor(e3_pdb, protein1_scores, args.output_dir / f"{args.e3}_MetaESI_colored.pdb")
        update_pdb_bfactor(sub_pdb, protein2_scores, args.output_dir / f"{args.candidate_sub}_MetaESI_colored.pdb")

        print(f"  • Annotated PDBs: {args.e3}_MetaESI_colored.pdb, {args.candidate_sub}_MetaESI_colored.pdb")

        print("\n" + "=" * 50)
        print(f"      Prediction completed successfully!")
        print("=" * 50)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    run(args)

