"""
Run Chai-1 for antibody structure prediction.

This script is based on the Chai-1 script from the Chai-1 repository:
https://github.com/chaidiscovery/chai-lab/

Attribution:
If you use this code, the Chai-1 model, or outputs produced by it in your research, please cite:

@article{Chai-1-Technical-Report,
        title        = {Chai-1: Decoding the molecular interactions of life},
        author       = {{Chai Discovery}},
        year         = 2024,
        journal      = {bioRxiv},
        publisher    = {Cold Spring Harbor Laboratory},
        doi          = {10.1101/2024.10.10.615955},
        url          = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955},
        elocation-id = {2024.10.10.615955},
        eprint       = {https://www.biorxiv.org/content/early/2024/10/11/2024.10.10.615955.full.pdf}
}

License:
- Chai-1 is released under an Apache 2.0 License (both code and model weights), which means it can be used for both academic and commercial purposes, including for drug discovery.
- See https://github.com/chaidiscovery/chai-lab/LICENSE for details.

"""

import shutil
from pathlib import Path
import numpy as np
from chai_lab.chai1 import run_inference
from Bio import PDB
import torch
import os
import hashlib
import time
import random
import pandas as pd


def generate_unique_hash():
    """
    Generate a unique hash identifier for temporary directories.

    Creates a collision-resistant hash using timestamp, process ID, and random number
    to ensure unique temporary directory names for concurrent Chai-1 runs.

    Returns:
        str: 16-character hexadecimal hash string.
    """
    timestamp = str(time.time())
    process_id = str(os.getpid())
    random_num = random.randint(0, 1000000)

    data_to_hash = f"{timestamp}-{process_id}-{random_num}"

    return hashlib.sha256(data_to_hash.encode()).hexdigest()[:16]


def run_chai(
    binder_sequence: str,
    output_dir: str,
    save_dir: str,
    pdb: str,
    target_chain: str = "A",
    seed: int = 0,
    cdr3_idx = None,
    hotspot_residue = None,
    binder_chain = "B",
    target_len: int = None,
):
    """
    Run Chai-1 structure prediction for antibody-target complex.

    Predicts the 3D structure of an antibody binder in complex with its target
    using the Chai-1 deep learning model. The function handles file preparation,
    model inference, result processing, and cleanup.

    Args:
        binder_sequence (str): Amino acid sequence of the antibody binder.
        output_dir (str): Temporary directory for intermediate files.
        save_dir (str): Final directory to save the predicted structure.
        pdb (str): Path to PDB file containing the target protein structure.
        target_chain (str, optional): Chain ID of target protein. Defaults to 'A'.
        seed (int, optional): Random seed for reproducible predictions. Defaults to 0.

    Returns:
        tuple: (structure_path, scores_dict) where:
            - structure_path (str): Path to predicted complex PDB file
            - scores_dict (dict): Confidence metrics and scores
    """
    # Create temporary directory structure for Chai-1 processing
    Path(os.path.join(output_dir, "tmp")).mkdir(exist_ok=True)
    # Generate unique identifier to avoid conflicts with concurrent runs
    hash_id = generate_unique_hash()

    # Create unique temporary directory (must not exist to avoid conflicts)
    Path(os.path.join(output_dir, "tmp", hash_id)).mkdir(exist_ok=False)
    fasta_path = Path(os.path.join(output_dir, "tmp", hash_id, "example.fasta"))

    # Extract protein sequences from input PDB file
    sequences = get_sequence_from_pdb(pdb)

    # Create FASTA input file with target and binder sequences
    target_chains = target_chain.split(",")
    with open(fasta_path, "w") as fh:
        for i, ch in enumerate(target_chains):
            fh.write(f">protein|name=target_protein_{i}\n")
            fh.write(sequences[ch] + "\n")
        fh.write(">protein|name=binder_protein\n")
        fh.write(binder_sequence + "\n")

    # Create empty output directory (required by Chai-1 inference)
    output_dir = Path(os.path.join(output_dir, "tmp", hash_id, "outputs"))
    output_dir.mkdir(exist_ok=False)

    constraint = False
    if cdr3_idx is not None and hotspot_residue is not None:
        constraint = True
        constraint_path = Path(__file__).with_name("chai.restraints")
        rests_df = pd.read_csv(constraint_path)
        rest_residue = binder_sequence[cdr3_idx]
        rests_df.loc[0, 'chainB'] = binder_chain
        rests_df.loc[0, 'res_idxB'] = f'{rest_residue}{cdr3_idx}'
        rests_df.loc[0, 'res_idxA'] = hotspot_residue
        rests_df.to_csv(constraint_path, index=False)

    # Run Chai-1 structure prediction with optimized parameters
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=3,  # Number of structure refinement cycles
        num_diffn_timesteps=200,  # Diffusion sampling steps
        device="cuda:0",  # GPU device for acceleration
        use_esm_embeddings=True,  # Use ESM language model embeddings
        seed=seed,  # Random seed for reproducibility
        constraint_path = constraint_path if constraint else None,  # Path to restraints file
    )

    # Convert output CIF files to PDB format for compatibility
    cif_paths = candidates.cif_paths

    pdb_paths = convert_cif_paths_to_pdb(cif_paths, hash_id)

    # Extract aggregate confidence scores from all predicted structures
    agg_scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]

    # Load detailed scores and select the best-scoring structure
    best_sample = agg_scores.index(max(agg_scores))
    scores = np.load(output_dir.joinpath(f"scores.model_idx_{best_sample}.npz"))
    pae_matrix, pae, plddt = (
        candidates.pae[best_sample],
        candidates.pae[best_sample],
        candidates.plddt[best_sample],
    )
    scores_dict = {key: scores[key] for key in scores.keys()}
    scores_dict["pae_matrix"] = pae_matrix
    scores_dict["agg_score"] = np.mean(agg_scores)
    scores_dict["pae"] = torch.mean(pae)
    scores_dict["plddt"] = torch.mean(plddt)
    scores_dict["plddt_binder"] = torch.mean(
        plddt[target_len:]
    )  # Average pLDDT for binder only
    scores_dict["binder_pae"] = torch.mean(
        torch.mean(pae[target_len:, target_len:])
    )  # Average PAE for binder-target interface
    scores_dict["chain_ptm"] = [1] #placeholder values for chai
    scores_dict["chain_iptm"] = [1] #placeholder values for chai

    # Copy best structure to final save directory and clean up
    pdb_path = pdb_paths[best_sample]

    os.makedirs(save_dir, exist_ok=True)

    shutil.copy(pdb_path, save_dir)
    new_path = os.path.join(save_dir, pdb_path.split("/")[-1])

    shutil.rmtree(os.path.join(output_dir, "tmp", hash_id), ignore_errors=True)

    return new_path, scores_dict


def get_sequence_from_pdb(pdb_file, file_format="pdb"):
    """
    Extract protein sequences from PDB/mmCIF structure files.

    Parses structural files to extract amino acid sequences for each protein chain.
    This is used to prepare input sequences for structure prediction.

    Args:
        pdb_file (str): Path to the PDB or mmCIF structure file.
        file_format (str, optional): File format ('pdb' or 'cif'). Defaults to 'pdb'.

    Returns:
        dict: Mapping of chain IDs to their amino acid sequences.
    """
    if file_format == "pdb":
        parser = PDB.PDBParser(QUIET=True)
    else:
        parser = PDB.MMCIFParser(QUIET=True)

    structure = parser.get_structure("protein", pdb_file)
    ppb = PDB.PPBuilder()

    sequences = {}
    for chain in structure.get_chains():
        chain_id = chain.id
        pp = ppb.build_peptides(chain)
        if pp:
            sequences[chain_id] = "".join(
                [str(peptide.get_sequence()) for peptide in pp]
            )

    return sequences


def convert_cif_paths_to_pdb(cif_paths, seed):
    """
    Convert Chai-1 output CIF files to PDB format.

    Chai-1 outputs structures in mmCIF format, but many downstream tools
    expect PDB format. This function converts all predicted structures
    and adds seed information to filenames for tracking.

    Args:
        cif_paths (List[str]): List of paths to CIF structure files.
        seed (int): Random seed used for prediction (added to filename).

    Returns:
        List[str]: List of paths to converted PDB files.
    """
    pdb_paths = []
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()

    for cif_path in cif_paths:
        # Generate output PDB filename with seed identifier
        pdb_path = str(Path(cif_path).with_suffix(".pdb"))
        # Add seed to filename for result tracking
        pdb_path = pdb_path.replace(".pdb", f"_{seed}.pdb")

        # Convert CIF to PDB format using Biopython
        structure = parser.get_structure("structure", cif_path)
        io.set_structure(structure)
        io.save(pdb_path)

        pdb_paths.append(pdb_path)

    return pdb_paths
