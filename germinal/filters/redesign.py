"""
AbMPNN redesign utilities for Germinal.

Attribution:
- The AbMPNN model weights (arXiv:2310.19513) were presented at the 2023 ICML Workshop on Computational Biology.
- Model weights and CSV files with the train, test, and validation splits across the SAbDab and ImmuneBuilder datasets are provided by the authors.
- AbMPNN is based on ProteinMPNN and can be run using the corresponding code: https://github.com/dauparas/ProteinMPNN.
"""

import os
import tempfile
import multiprocessing as mp
import pickle
from typing import Dict, List, Any, Tuple
from germinal.utils.utils import hotspot_residues, clear_memory, get_sequence_from_pdb
from colabdesign.mpnn import mk_mpnn_model

mp.set_start_method("spawn", force=True)


def abmpnn_design(
    trajectory_pdb: str,
    trajectory_interface_residues: str,
    run_settings: Dict[str, Any],
    target_chain: str = "A",
    binder_chain: str = "B",
) -> Dict[str, Any]:
    """
    Generate redesigned sequences using AbMPNN for a given PDB structure.

    Args:
        trajectory_pdb: Path to the input PDB file.
        target_chain: Target chain identifier (e.g., 'A').
        binder_chain: Binder chain identifier (e.g., 'B').
        trajectory_interface_residues: Comma-separated string of interface residue indices.
        run_settings: Dictionary of settings.

    Returns:
        Dictionary containing redesigned sequence and associated scores.
    """
    abmpnn_model = mk_mpnn_model(
        backbone_noise=run_settings["backbone_noise"],
        model_name=run_settings["model_path"],
        weights=run_settings["mpnn_weights"],
    )

    # Determine which residues to fix during redesign
    design_chains = f"{target_chain},{binder_chain}"
    if run_settings.get("mpnn_fix_interface", False):
        fixed_positions = ",".join(target_chain.split(",") + [str(trajectory_interface_residues)]).rstrip(",")
    else:
        fixed_positions = target_chain

    # Prepare AbMPNN model inputs
    abmpnn_model.prep_inputs(
        pdb_filename=trajectory_pdb,
        chain=design_chains,
        fix_pos=fixed_positions,
        rm_aa=run_settings["omit_AAs"],
    )

    # Sample redesigned sequences
    abmpnn_sequences = abmpnn_model.sample(
        temperature=run_settings["sampling_temp"], num=1, batch=run_settings["num_seqs"]
    )

    # Clean up memory
    del abmpnn_model
    clear_memory()

    return abmpnn_sequences


def abmpnn_worker(
    trajectory_pdb: str,
    target_chain: str,
    binder_chain: str,
    trajectory_interface_residues: str,
    run_settings: Dict[str, Any],
    output_path: str,
) -> None:
    """
    Worker function for AbMPNN sequence generation in multiprocessing.

    Args:
        trajectory_pdb: Path to the trajectory PDB file
        target_chain: Target chain identifier
        binder_chain: Binder chain identifier
        trajectory_interface_residues: Interface residues to fix
        run_settings: Dictionary containing AbMPNN settings
        output_path: Path to save the output pickle file
    """
    result = abmpnn_design(
        trajectory_pdb,
        trajectory_interface_residues,
        run_settings,
        target_chain = target_chain,
        binder_chain = binder_chain,
    )
    with open(output_path, "wb") as f:
        pickle.dump(result, f)


def get_abmpnn_sequences(
    trajectory_pdb_af: str,
    run_settings: Dict[str, Any],
    cdr_positions: List[int],
    atom_distance_cutoff: float = 3.0,
    target_chain: str = "A",
    binder_chain: str = "B",
) -> List[Dict[str, Any]]:
    """
    Generate AbMPNN redesigned sequences for a given trajectory.

    Args:
        trajectory_pdb_af: Path to the trajectory PDB file from AF2/ColabDesign
        target_chain: Target chain identifier (e.g., 'A')
        binder_chain: Binder chain identifier (e.g., 'B')
        run_settings: Dictionary containing run settings including:
            - max_mpnn_sequences: Maximum number of MPNN sequences to return
            - cdr_positions: CDR positions to redesign
        atom_distance_cutoff: Distance cutoff for interface residue detection

    Returns:
        List of dictionaries containing MPNN sequences, each with:
            - seq: The redesigned sequence
            - score: MPNN score for the sequence
            - seqid: Sequence identity to original
    """
    # Get interface residues
    trajectory_interface_residues = hotspot_residues(
        trajectory_pdb_af,
        binder_chain,
        target_chain=target_chain,
        atom_distance_cutoff=atom_distance_cutoff,
    )

    # Convert interface residues to PDB IDs format
    interface_residues_pdb_ids = []
    for pdb_res_num, _ in trajectory_interface_residues.items():
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")

    # Determine residues to fix (all non-CDR positions + interface residues)
    chains_pdb = get_sequence_from_pdb(trajectory_pdb_af)
    if len(chains_pdb) < 3: # Using PDB from AFM which has only 2 chains
        binder_chain = "B"
    length = len(chains_pdb[binder_chain])
    residues_to_fix = [
        f"{binder_chain}{pos+1}"
        for pos in range(0, length)
        if pos not in run_settings["cdr_positions"]
    ]
    residues_to_fix = set(residues_to_fix + interface_residues_pdb_ids)
    residues_to_fix = ",".join(residues_to_fix)

    # Run MPNN in a separate process to avoid memory issues
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        output_path = tf.name

    proc = mp.Process(
        target=abmpnn_worker,
        args=(
            trajectory_pdb_af,
            target_chain,
            binder_chain,
            residues_to_fix,
            run_settings,
            output_path,
        ),
    )
    proc.start()
    proc.join()

    # Read result from file
    try:
        with open(output_path, "rb") as f:
            abmpnn_trajectories = pickle.load(f)
        os.unlink(output_path)  # Clean up temporary file
    except Exception as e:
        print(f"Error reading AbMPNN results: {e}")
        if os.path.exists(output_path):
            os.unlink(output_path)
        return []

    # Process and deduplicate MPNN sequences
    if not abmpnn_trajectories or "seq" not in abmpnn_trajectories:
        print("No MPNN sequences generated")
        return []

    # Create unique sequences dictionary and sort by score
    unique_sequences = {}
    for n in range(len(abmpnn_trajectories["seq"])):
        seq = abmpnn_trajectories["seq"][n][-length:]  # Take only binder sequence
        if seq not in unique_sequences:
            unique_sequences[seq] = {
                "seq": seq,
                "score": abmpnn_trajectories["score"][n],
                "seqid": abmpnn_trajectories["seqid"][n],
            }

    # Sort by AbMPNN score (lower is better) and limit to max sequences
    abmpnn_sequences = sorted(unique_sequences.values(), key=lambda x: x["score"])
    max_sequences = run_settings.get("max_mpnn_sequences", 4)
    abmpnn_sequences = abmpnn_sequences[:max_sequences]

    print(f"Generated {len(abmpnn_sequences)} unique AbMPNN sequences")
    for i, seq_data in enumerate(abmpnn_sequences):
        print(
            f"  Sequence {i + 1}: score={seq_data['score']:.3f}, seqid={seq_data['seqid']:.3f}"
        )

    return abmpnn_sequences


def run_abmpnn_redesign_pipeline(
    trajectory_pdb_af: str,
    target_chain: str,
    binder_chain: str,
    run_settings: Dict[str, Any],
    atom_distance_cutoff: float = 3.0,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Complete AbMPNN redesign pipeline for a trajectory.

    Args:
        trajectory_pdb_af: Path to the trajectory PDB file from AF2/ColabDesign
        target_chain: Target chain identifier (e.g., 'A')
        binder_chain: Binder chain identifier (e.g., 'B')
        run_settings: Dictionary containing run settings
        atom_distance_cutoff: Distance cutoff for interface residue detection

    Returns:
        Tuple of (abmpnn_sequences, success_flag)
            - abmpnn_sequences: List of AbMPNN redesigned sequences
            - success_flag: Boolean indicating if redesign was successful
    """
    try:
        abmpnn_sequences = get_abmpnn_sequences(
            trajectory_pdb_af=trajectory_pdb_af,
            run_settings=run_settings,
            cdr_positions=run_settings["cdr_positions"],
            atom_distance_cutoff=atom_distance_cutoff,
        )

        success = len(abmpnn_sequences) > 0
        if not success:
            print("AbMPNN redesign failed: no sequences generated")

        return abmpnn_sequences, success

    except Exception as e:
        print(f"AbMPNN redesign pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        return [], False
