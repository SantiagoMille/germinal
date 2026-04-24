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

    On exception, dumps the full traceback to ``{output_path}.err`` so the
    parent can read it and raise a loud error. Without this, child failures
    only surface as a missing/empty pickle with no diagnostics.

    Args:
        trajectory_pdb: Path to the trajectory PDB file
        target_chain: Target chain identifier
        binder_chain: Binder chain identifier
        trajectory_interface_residues: Interface residues to fix
        run_settings: Dictionary containing AbMPNN settings
        output_path: Path to save the output pickle file
    """
    import traceback
    try:
        result = abmpnn_design(
            trajectory_pdb,
            trajectory_interface_residues,
            run_settings,
            target_chain = target_chain,
            binder_chain = binder_chain,
        )
        with open(output_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        err_path = f"{output_path}.err"
        try:
            with open(err_path, "w") as fh:
                fh.write(
                    f"abmpnn_worker failed for trajectory_pdb={trajectory_pdb}\n\n"
                )
                fh.write(traceback.format_exc())
        except Exception:
            pass
        raise


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
    chains_pdb = get_sequence_from_pdb(trajectory_pdb_af)
    if len(chains_pdb) < 3:  # Using PDB from AFM which has only 2 chains
        binder_chain = "B"
    length = len(chains_pdb[binder_chain])
    # Always fix framework (non-CDR) positions
    residues_to_fix = [
        f"{binder_chain}{pos+1}"
        for pos in range(0, length)
        if pos not in run_settings["cdr_positions"]
    ]

    if atom_distance_cutoff > 0.0:
        # Also fix interface CDR residues (preserve binding contacts)
        trajectory_interface_residues = hotspot_residues(
            trajectory_pdb_af,
            binder_chain,
            target_chain=target_chain,
            atom_distance_cutoff=atom_distance_cutoff,
        )
        interface_residues_pdb_ids = [
            f"{binder_chain}{pdb_res_num}"
            for pdb_res_num in trajectory_interface_residues.keys()
        ]
        residues_to_fix = set(residues_to_fix + interface_residues_pdb_ids)
    else:
        residues_to_fix = set(residues_to_fix)

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

    # If the child crashed, surface its traceback loudly and raise so the
    # caller stops the trajectory. Silently returning [] previously hid
    # AbMPNN bugs as "no sequences generated".
    if proc.exitcode != 0:
        err_path = f"{output_path}.err"
        traceback_text = ""
        if os.path.exists(err_path):
            try:
                with open(err_path) as fh:
                    traceback_text = fh.read()
                os.unlink(err_path)
            except Exception:
                pass
        if os.path.exists(output_path):
            os.unlink(output_path)
        msg = (
            f"\n{'=' * 78}\n"
            f"[ABMPNN ERROR] worker process exited non-zero "
            f"(exitcode={proc.exitcode}) for trajectory_pdb={trajectory_pdb_af}\n"
            f"{'=' * 78}\n"
            f"{traceback_text or '(no traceback captured — child may have died via signal)'}"
            f"{'=' * 78}\n"
        )
        print(msg, flush=True)
        raise RuntimeError(
            f"abmpnn_worker failed (exitcode={proc.exitcode}); see traceback above."
        )

    # Read result from file
    try:
        with open(output_path, "rb") as f:
            abmpnn_trajectories = pickle.load(f)
        os.unlink(output_path)
    except Exception as e:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError(
            f"abmpnn_worker exited 0 but produced unreadable pickle at "
            f"{output_path}: {e}"
        )

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
    run_settings: Dict[str, Any],
    atom_distance_cutoff: float = 3.0,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Complete AbMPNN redesign pipeline for a trajectory.

    Note: target_chain / binder_chain are NOT parameters here — they were
    historically accepted but never forwarded to get_abmpnn_sequences (which
    uses its own defaults of A/B). Removed from the signature to avoid the
    misleading impression that the wrapper honors them.

    Args:
        trajectory_pdb_af: Path to the trajectory PDB file from AF2/ColabDesign
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

    except Exception:
        # Fail-fast: an AbMPNN failure here means either OOM, a config bug, or
        # genuine model breakage — none of which are safe to silently skip.
        # Previous behavior (return [], False) made the trajectory loop continue
        # to the next seed, hiding real bugs and wasting GPU time. Re-raise so
        # SLURM marks the whole job FAILED and the user investigates.
        import traceback
        print(
            f"\n{'=' * 78}\n"
            f"[ABMPNN PIPELINE FATAL] redesign failed; aborting the run "
            f"so the underlying issue can be diagnosed.\n"
            f"{'=' * 78}",
            flush=True,
        )
        traceback.print_exc()
        raise
