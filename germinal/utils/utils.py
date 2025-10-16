"""Germinal Utility Functions Module

This module provides a comprehensive collection of utility functions for the Germinal
protein design system. It includes functions for device management, time formatting,
PDB file processing, structural analysis, secondary structure calculation, and various
protein manipulation operations.

The module implements:
- Device detection and validation (JAX/CUDA)
- Time and timestamp utilities
- PDB file cleaning and manipulation
- Memory management for GPU operations
- Protein structure analysis and hotspot identification
- CDR position computation and framework handling
- Secondary structure analysis with DSSP
- Clash detection and structural validation
- Chain manipulation and complex assembly

Key Function Categories:
    Time/Device Utilities:
        get_clean_time, get_timestamp, get_torch_device, get_jax_device
    
    Memory Management:
        clear_memory
        
    PDB Processing:
        clean_pdb, get_sequence_from_pdb, create_starting_structure
        
    Structural Analysis:
        hotspot_residues, calculate_clash_score, calc_ss_percentage
        
    Chain Manipulation:
        split_dimer_chain_a, merge_chains_with_offset
        
    CDR/Framework Processing:
        compute_cdr_positions, idx_from_ranges

Dependencies:
    - JAX for GPU device detection
    - PyTorch for CUDA validation
    - BioPython for PDB processing and structure analysis
    - NumPy/SciPy for numerical computations
    - DSSP for secondary structure calculation

"""

import time
import gc
from typing import Optional
from collections import defaultdict
import jax
import torch
import numpy as np
from numpy.random import default_rng
from scipy.spatial import cKDTree
from Bio import PDB
from Bio.PDB import (
    PDBParser,
    DSSP,
    Selection,
)

from Bio.SeqUtils import seq1

def get_clean_time(end_time: float, start_time: float) -> str:
    """Convert elapsed time between timestamps to human-readable format.
    
    Calculates the elapsed time between two timestamps and formats it as
    a human-readable string with hours, minutes, and seconds.
    
    Args:
        end_time (float): End timestamp (typically from time.time())
        start_time (float): Start timestamp (typically from time.time())
        
    Returns:
        str: Formatted time string in format "Xh Ym Zs"
    """
    elapsed_time = end_time - start_time
    hh, remainder = divmod(elapsed_time, 3600)
    mm, ss = divmod(remainder, 60)
    return f"{int(hh)}h {int(mm)}m {int(ss)}s"


def get_timestamp() -> str:
    """Generate timestamp string for unique run identification.
    
    Creates a timestamp string in YYYYMMDD_HHMMSS format suitable for
    creating unique identifiers for design runs and file naming.
    
    Returns:
        str: Timestamp string in format "YYYYMMDD_HHMMSS"
    """
    return time.strftime("%Y%m%d_%H%M%S")


def make_rng(seed: Optional[int] = None):
    """Create a NumPy random number generator with optional seed.
    
    Creates a new NumPy random number generator instance, optionally
    seeded for reproducible random number generation.
    
    Args:
        seed (Optional[int]): Random seed for reproducible results.
            If None, generator will be unseeded.
            
    Returns:
        numpy.random.Generator: NumPy random number generator instance
    """
    return default_rng(seed)


def get_torch_device() -> str:
    """Detect and return available PyTorch device.
    
    Checks for CUDA availability and returns the appropriate device
    string for PyTorch operations.
    
    Returns:
        str: 'cuda' if CUDA is available, 'cpu' otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def clean_pdb(pdb_file):
    """Remove unnecessary information from PDB files for clean processing.
    
    Filters PDB files to retain only essential structural information by
    keeping ATOM, HETATM, MODEL, TER, and END records while removing
    Rosetta-specific annotations and other metadata.
    
    Args:
        pdb_file (str): Path to the PDB file to be cleaned
    """
    # Read the pdb file and filter relevant lines
    with open(pdb_file, "r") as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(("ATOM", "HETATM", "MODEL", "TER", "END"))
        ]

    # Write the cleaned lines back to the original pdb file
    with open(pdb_file, "w") as f_out:
        f_out.writelines(relevant_lines)


def clear_memory(clear_jax: bool = True):
    """Clear GPU and system memory caches for optimal performance.
    
    Performs comprehensive memory cleanup including Python garbage collection,
    PyTorch CUDA cache clearing, and JAX cache clearing. Handles cases where
    GPU libraries may not be available gracefully.
    """
    gc.collect()

    try:
        torch.cuda.empty_cache()
    except Exception:
        print("Warning: Torch cache not cleared because it is not available")
        pass

    try:
        if clear_jax:
            jax.clear_caches()
    except Exception:
        print("Warning: JAX cache not cleared because it is not available")
        pass

    gc.collect()


def get_jax_device():
    """Detect and verify JAX GPU device availability.
    
    Checks for JAX GPU devices and returns available GPU device list.
    Provides comprehensive GPU availability validation for JAX operations.
    
    Returns:
        list or None: List of available JAX GPU devices if found,
            None if no GPU devices available or JAX not accessible
            
    Raises:
        ValueError: If no JAX GPU devices are detected
    """

    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == "gpu"]
        if gpu_devices:
            print(f"JAX GPU devices available: {gpu_devices}")
            return gpu_devices
        else:
            raise ValueError("No JAX GPU devices found")
    except Exception:
        print("Warning: JAX not available")
        return None


def copy_dict(d: dict) -> dict:
    """Create a deep copy of a dictionary with all nested structures.
    
    Performs deep copying of a dictionary to ensure complete independence
    from the original, including all nested dictionaries, lists, and objects.
    
    Args:
        d (dict): Dictionary to be deep copied
        
    Returns:
        dict: Independent deep copy of the input dictionary
    """
    import copy

    return copy.deepcopy(d)


def idx_from_ranges(ranges, chain="B", offset=0):
    """Convert range string specification to zero-based index list.
    
    Parses a range string (e.g., 'B1-10,A5') into a list of zero-based indices,
    with support for chain filtering and offset application. Handles both single
    residues and ranges with flexible chain specification.
    
    Args:
        ranges (str): Range specification string with format like 'B1-10,A5'
            or '1-10,15' (with or without chain prefixes)
        chain (str, optional): Chain identifier to filter for. Defaults to 'B'.
        offset (int, optional): Offset to apply to residue numbers. Defaults to 0.
        
    Returns:
        list: Zero-based indices corresponding to the specified ranges
    """
    rows = []
    ranges = ranges.replace(chain, "")
    for part in ranges.split(","):
        if part[0].isalpha():
            continue
        else:
            if "-" in part:
                start, end = map(int, part.split("-"))
                rows.extend(range(start + offset - 1, end + offset))
            else:
                rows.append(int(part) + offset - 1)
    return rows


def compute_cdr_positions(
    cdr_lengths: list[int], framework_lengths: list[int]
) -> list[int]:
    """Compute CDR residue positions from framework and CDR length specifications.
    
    Calculates the absolute residue positions for all CDRs based on framework
    region lengths and CDR lengths. Uses cumulative positioning to account for
    the sequential arrangement of framework and CDR regions.
    
    Args:
        cdr_lengths (list[int]): List of CDR lengths in sequential order
        framework_lengths (list[int]): List of framework region lengths
            corresponding to each CDR position
            
    Returns:
        list[int]: Flat list of all CDR residue positions (zero-based indices)
    """

    cumulative = 0
    positions = []
    for i, cdr_length in enumerate(cdr_lengths):
        fw_len = framework_lengths[i] + cumulative
        positions.extend(range(fw_len, fw_len + cdr_length))
        cumulative = fw_len + cdr_length
    return positions


######### BIOPYTHON UTILS #########


def get_sequence_from_pdb(pdb_path: str) -> dict:
    """Extract protein sequences from PDB file organized by chain.
    
    Parses a PDB file and extracts the amino acid sequence for each chain,
    converting three-letter amino acid codes to single-letter codes.
    
    Args:
        pdb_path (str): Path to the PDB file to process
        
    Returns:
        dict: Dictionary mapping chain identifiers to their amino acid sequences
            in single-letter code format
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_path)
    chains = {
        chain.id: seq1("".join(residue.resname for residue in chain))
        for chain in structure.get_chains()
    }
    return chains


def create_starting_structure(
    save_path,
    binder_pdb,
    target_pdb,
    binder_chain="B",
    start_binder_chain="A",
    target_chain="A",
):
    """Create combined PDB structure from separate binder and target files.
    
    Combines a binder PDB and target PDB into a single complex structure,
    handling chain renaming and multi-chain targets. Creates the starting
    structure needed for protein design optimization.
    
    Args:
        save_path (str): Path where the combined structure will be saved
        binder_pdb (str): Path to the binder PDB structure
        target_pdb (str): Path to the target PDB structure
        binder_chain (str, optional): Chain ID for binder in output. Defaults to 'B'.
        start_binder_chain (str, optional): Chain ID of binder in input file. Defaults to 'A'.
        target_chain (str, optional): Chain ID(s) for target (comma-separated for multiple). Defaults to 'A'.
        
    Returns:
        str: Amino acid sequence of the binder chain in single-letter code
    """
    target_chain = target_chain.split(",")

    # Set up the parser and structure objects
    parser = PDB.PDBParser(QUIET=True)
    # Load the two structures
    structure1 = parser.get_structure("structure1", target_pdb)
    structure2 = parser.get_structure("structure2", binder_pdb)
    # Create a new structure to hold the combined molecules
    combined = PDB.Structure.Structure("combined")
    # Create a model in the new structure
    model = PDB.Model.Model(0)
    combined.add(model)
    # Add chain A from first structure
    chainA = structure1[0][target_chain[0]]
    model.add(chainA)
    # Add chain from second structure as chain B
    chainB = structure2[0][start_binder_chain]
    chainB.id = binder_chain  # Rename the chain to B

    offset = np.array([30.0, 30.0, 0.0])  # move binder +30 Å along X
    for atom in chainB.get_atoms():
        atom.set_coord(atom.coord + offset)

    model.add(chainB)
    for chain_id in target_chain[1:]:
        chain = structure1[0][chain_id]
        chain.id = chain_id
        model.add(chain)
    # Save the combined structure
    io = PDB.PDBIO()
    io.set_structure(combined)
    io.save(save_path)
    print(f"Created starting PDB at: {save_path}")

    return get_sequence_from_pdb(save_path)[binder_chain]


three_to_one_map = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def hotspot_residues(
    trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0, target_chain="A"
):
    """Identify interface hotspot residues between binder and target.
    
    Analyzes a protein complex structure to identify binder residues that
    are in close contact with the target protein, defining the binding interface.
    Uses spatial proximity analysis with KD-trees for efficient computation.
    
    Args:
        trajectory_pdb (str): Path to the PDB file containing the protein complex
        binder_chain (str, optional): Chain identifier for the binder protein. Defaults to 'B'.
        atom_distance_cutoff (float, optional): Maximum distance (Å) for interface contacts. Defaults to 4.0.
        target_chain (str, optional): Chain identifier for the target protein. Defaults to 'A'.
        
    Returns:
        dict: Dictionary mapping binder residue numbers to single-letter amino acid codes
            for all residues with atoms within the distance cutoff of the target
    """
    # Parse the PDB file
    target_chain = target_chain.split(",") if isinstance(target_chain, str) else target_chain
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # AFM generates a single target chain even for multichain targets so need to rename binder as "B"
    model = next(structure.get_models())
    chains = list(model.get_chains())
    if len(chains) == 2:
        binder_chain = 'B'

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], "A")
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = []
    for chain_id in target_chain:
        target_atoms.extend(Selection.unfold_entities(structure[0][chain_id], "A"))
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues


def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    """Calculate structural clash score for protein structure validation.
    
    Analyzes a protein structure to identify and count atomic clashes based on
    distance thresholds. Provides options for CA-only analysis or full atomic
    clash detection with proper exclusions for bonded and sequential residues.
    
    Args:
        pdb_file (str): Path to the PDB file to analyze
        threshold (float, optional): Distance threshold (Å) below which atoms are
            considered clashing. Defaults to 2.4.
        only_ca (bool, optional): If True, only analyze CA-CA distances.
            If False, analyze all heavy atoms. Defaults to False.
            
    Returns:
        int: Number of clashing atom pairs found in the structure
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == "H":  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != "CA":
                        continue
                    atoms.append(atom.coord)
                    atom_info.append(
                        (chain.id, residue.id[1], atom.get_name(), atom.coord)
                    )

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for i, j in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)


def get_binder_struct(pdb_path, binder_chain="B"):
    """Extract and save binder chain as separate PDB structure.
    
    Isolates the binder chain from a protein complex and saves it as a
    separate PDB file for individual analysis or processing.
    
    Args:
        pdb_path (str): Path to the input PDB file containing the complex
        binder_chain (str, optional): Chain identifier for the binder. Defaults to 'B'.
        
    Returns:
        str: Path to the saved binder-only PDB file
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    chain = structure[0][binder_chain]
    model = PDB.Model.Model(0)
    model.add(chain)
    binder_pdb = PDB.Structure.Structure("binder")
    binder_pdb.add(model)
    save_path = pdb_path.replace(".pdb", "_binder.pdb")
    io = PDB.PDBIO()
    io.set_structure(binder_pdb)
    io.save(save_path)

    return save_path


def split_dimer_chain_a(pdb_path, gap_threshold=50):
    """Split chain A at large gaps to separate dimer components.
    
    Analyzes chain A for large gaps in residue numbering and splits it into
    separate chains when gaps exceed the threshold. This is useful for handling
    dimeric targets where both chains are labeled as chain A with a gap.
    
    The function performs the following operations:
    1. Parses the PDB structure (single model assumed)
    2. Analyzes chain A for gaps ≥ gap_threshold in residue numbering
    3. If gap found: splits chain A into chain A (first part) and chain C (second part)
    4. If no gap: reindexes chain A residues starting from 1
    5. Preserves chain B (binder) unchanged
    6. Writes new structure with chain order: A → B → C
    
    Args:
        pdb_path (str): Path to the PDB file to modify (modified in-place)
        gap_threshold (int, optional): Residue gap size that triggers chain split. Defaults to 50.
        
    Returns:
        str: The same pdb_path (file modified in-place)
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    model = structure[0]  # Assume a single model

    # Collect references to chain A and chain B (the nanobody).
    chainA = None
    chainB = None
    for ch in model:
        if ch.id == "A":
            chainA = ch
        elif ch.id == "B":
            chainB = ch

    # We will build these segments as new Chain objects:
    new_chainA = None  # Will hold the first part (or all) of Chain A
    new_chainC = None  # Will hold the second part of Chain A, if there's a gap
    new_chainB = None  # Will hold Chain B

    #############################
    # 1) Handle Chain A
    #############################
    if chainA is not None:
        residues = list(chainA)
        if residues:
            # Look for a gap
            res_nums = [res.id[1] for res in residues]
            gaps = np.diff(res_nums)
            split_points = np.where(gaps >= gap_threshold)[0]

            if len(split_points) > 0:
                # Take the first gap only (common scenario)
                split_idx = split_points[0]
                # Part1 = up to (and including) split_idx
                A_part1 = residues[: split_idx + 1]
                # Part2 = after split_idx
                A_part2 = residues[split_idx + 1 :]

                # Reindex them
                new_chainA = copy_residues_with_reindex(A_part1, new_chain_id="A")
                new_chainC = copy_residues_with_reindex(A_part2, new_chain_id="C")
                # print(f"Split Chain A at residue {A_part1[-1].id[1]}")
            else:
                # No gap => entire chain A (reindexed) remains chain A
                new_chainA = copy_residues_with_reindex(residues, new_chain_id="A")

    #############################
    # 2) Handle Chain B (nanobody)
    #############################
    if chainB is not None:
        # If you do NOT want to change numbering in B, just copy it:
        new_chainB = chainB.copy()
        new_chainB.id = "B"  # Ensure it remains B

        # If you do want to reset numbering, do:
        # new_chainB = copy_residues_with_reindex(chainB, new_chain_id="B")

    #############################
    # 3) Build a new structure in the desired order: A → B → C
    #############################
    new_structure = PDB.Structure.Structure("new_structure")
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)

    # Add chain A first (if it exists)
    if new_chainA is not None:
        new_model.add(new_chainA)

    # Then chain B
    if new_chainB is not None:
        new_model.add(new_chainB)
    # print(new_chainC)
    # Finally chain C (the second half of A if a gap was found)
    if new_chainC is not None:
        new_model.add(new_chainC)
    # print(f'num residues in chain C: {len(new_chainC)}')
    print(f"Length of new model: {len(new_model)}")
    # Write the new structure back to pdb_path in place
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(pdb_path)

    return pdb_path


def copy_residues_with_reindex(residues, new_chain_id="A"):
    """Copy residues to new chain with sequential renumbering.
    
    Creates a new chain containing copies of the input residues with
    residue numbers reset to sequential values starting from 1.
    
    Args:
        residues (list): List of BioPython Residue objects to copy
        new_chain_id (str, optional): Chain identifier for the new chain. Defaults to 'A'.
        
    Returns:
        Bio.PDB.Chain: New chain containing copied residues with renumbered IDs
    """
    from Bio.PDB import Chain

    new_chain = Chain.Chain(new_chain_id)

    for i, old_res in enumerate(residues, start=1):
        new_res = old_res.copy()
        # Reset the residue ID to (hetero_flag, new_resnum, insertion_code)
        new_res.id = (" ", i, " ")
        new_chain.add(new_res)

    return new_chain


def merge_chains_with_offset(pdb_path, output_path=None, offset=51):
    """
    Note: this is claude generated and I only tested for vegf...

    Merge chain C into chain A with a specified residue offset.

    Parameters:
    pdb_path (str): Path to the input PDB file
    output_path (str, optional): Path for the output PDB file. If None, will use input name + '_merged.pdb'
    offset (int): Residue number offset to apply when merging chain C into chain A

    Returns:
    str: Path to the output PDB file
    """
    # Parse PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)

    # Create new structure for output
    new_structure = PDB.Structure.Structure("new")
    new_model = PDB.Model.Model(0)
    new_structure.add(new_model)

    # Find the highest residue number in chain A to determine where to start chain C
    chain_a = None
    chain_c = None

    for chain in structure[0]:
        if chain.id == "A":
            chain_a = chain
        elif chain.id == "C":
            chain_c = chain

    if chain_a is None:
        raise ValueError("Chain A not found in the structure")
    if chain_c is None:
        raise ValueError("Chain C not found in the structure")

    # Find max residue number in chain A
    max_res_a = max([res.id[1] for res in chain_a])
    start_res_c = max_res_a + offset

    # Create a new chain A that will contain both original chains
    new_chain_a = PDB.Chain.Chain("A")

    # Add all residues from chain A
    for res in chain_a:
        new_chain_a.add(res.copy())

    # Add all residues from chain C with new residue numbers
    for res in chain_c:
        old_res_id = res.id
        new_res_num = start_res_c + (res.id[1] - min([r.id[1] for r in chain_c]))
        # Create new residue ID (insertion code and hetero-flag remain the same)
        new_res_id = (old_res_id[0], new_res_num, old_res_id[2])

        # Need to create a new residue with the updated ID
        new_res = PDB.Residue.Residue(new_res_id, res.resname, res.segid)

        # Copy all atoms from the original residue
        for atom in res:
            new_res.add(atom.copy())

        # Add the new residue to chain A
        new_chain_a.add(new_res)

    # Add all other chains except C to the new model
    new_model.add(new_chain_a)
    for chain in structure[0]:
        if chain.id != "A" and chain.id != "C":
            new_model.add(chain.copy())

    # Save new structure to a temporary file
    if output_path is None:
        output_path = pdb_path

    io = PDB.PDBIO()
    io.set_structure(new_structure)
    temp_output = output_path + ".temp"
    io.save(temp_output)

    # Now we need to manually insert a TER record between the chains
    with open(temp_output, "r") as infile, open(output_path, "w") as outfile:
        lines = infile.readlines()
        modified_lines = []

        # Find the last atom of original chain A and the first atom of what was chain C
        last_a_idx = -1
        first_c_idx = -1

        for i, line in enumerate(lines):
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21]
                res_num = int(line[22:26].strip())

                if chain_id == "A" and res_num == max_res_a:
                    last_a_idx = i
                elif chain_id == "A" and res_num >= start_res_c and first_c_idx == -1:
                    first_c_idx = i

        if last_a_idx >= 0 and first_c_idx > last_a_idx:
            # Get the last atom line of chain A
            last_a_line = lines[last_a_idx]

            # Generate TER line
            atom_serial = int(last_a_line[6:11]) + 1
            res_name = last_a_line[17:20]
            chain_id = last_a_line[21]
            res_num = int(last_a_line[22:26])
            res_num_str = str(res_num).rjust(4)

            ter_line = f"TER   {str(atom_serial).rjust(5)}      {res_name} {chain_id}{res_num_str}                                                      \n"

            # Insert TER line after the last atom of chain A
            modified_lines = lines[: last_a_idx + 1] + [ter_line] + lines[first_c_idx:]
        else:
            modified_lines = lines

        # Write all lines to output file
        outfile.writelines(modified_lines)

    import os

    os.remove(temp_output)

    print(f"Saved merged structure to {output_path}")
    return output_path


def calc_ss_percentage(
    pdb_file,
    advanced_settings,
    chain_id="B",
    atom_distance_cutoff=4.0,
    return_dict=False,
    target_chain="A",
):
    """Calculate secondary structure percentages with interface analysis.
    
    Analyzes protein secondary structure using DSSP and calculates percentages
    of helix, sheet, and loop content for both the entire chain and interface
    residues. Also computes confidence metrics (pLDDT) for different regions.
    
    Args:
        pdb_file (str): Path to the PDB file for analysis
        advanced_settings (dict): Settings dictionary containing 'dssp_path' key
        chain_id (str, optional): Chain to analyze. Defaults to 'B'.
        atom_distance_cutoff (float, optional): Distance cutoff (Å) for interface
            definition. Defaults to 4.0.
        return_dict (bool, optional): If True, return dict format. If False,
            return tuple format. Defaults to False.
        target_chain (str, optional): Target chain for interface calculation. Defaults to 'A'.
        
    Returns:
        dict or tuple: Secondary structure percentages and confidence metrics.
            If return_dict=True: Dictionary with keys 'alpha_', 'beta_', 'loops_',
                'alpha_i', 'beta_i', 'loops_i', 'i_plddt', 'ss_plddt'
            If return_dict=False: Tuple (alpha%, beta%, loop%, alpha_i%, beta_i%,
                loop_i%, i_plddt, ss_plddt)
    """
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(
        hotspot_residues(
            pdb_file, chain_id, atom_distance_cutoff, target_chain=target_chain
        ).keys()
    )

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = "loop"
            if ss in ["H", "G", "I"]:
                ss_type = "helix"
            elif ss == "E":
                ss_type = "sheet"

            ss_counts[ss_type] += 1

            if ss_type != "loop":
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(
        total_residues, ss_counts["helix"], ss_counts["sheet"]
    )
    interface_percentages = calculate_percentages(
        total_interface_residues,
        ss_interface_counts["helix"],
        ss_interface_counts["sheet"],
    )

    i_plddt = (
        round(sum(plddts_interface) / len(plddts_interface) / 100, 2)
        if plddts_interface
        else 0
    )
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    if return_dict:
        return {
            "alpha_": percentages[0],
            "beta_": percentages[1],
            "loops_": percentages[2],
            "alpha_i": interface_percentages[0],
            "beta_i": interface_percentages[1],
            "loops_i": interface_percentages[2],
            "i_plddt": i_plddt,
            "ss_plddt": ss_plddt,
        }
    else:
        return (*percentages, *interface_percentages, i_plddt, ss_plddt)


def calculate_percentages(total, helix, sheet):
    """Calculate secondary structure percentages from counts.
    
    Converts raw counts of secondary structure elements to percentages
    with proper handling of zero totals.
    
    Args:
        total (int): Total number of residues
        helix (int): Number of helical residues
        sheet (int): Number of sheet residues
        
    Returns:
        tuple: (helix_percentage, sheet_percentage, loop_percentage) rounded to 2 decimals
    """
    helix_percentage = round((helix / total) * 100, 2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100, 2) if total > 0 else 0
    loop_percentage = (
        round(((total - helix - sheet) / total) * 100, 2) if total > 0 else 0
    )

    return helix_percentage, sheet_percentage, loop_percentage


def interface_cdrs(interface: str, cdrs: str, cdr3: str, binder_chain="B"):
    """Calculate CDR involvement in binding interface.
    
    Analyzes the overlap between interface residues and CDR regions to
    determine what fraction of the interface is composed of CDR residues.
    
    Args:
        interface (str): Interface residue specification string
        cdrs (str): All CDR residue positions
        cdr3 (str): CDR3 residue positions specifically
        
    Returns:
        tuple: (total_cdr_fraction, cdr3_fraction) where:
            - total_cdr_fraction: Fraction of interface residues that are CDRs
            - cdr3_fraction: Fraction of interface residues that are CDR3
    """
    interface = idx_from_ranges(interface, chain=binder_chain)
    # cdrs = idx_from_ranges(cdrs)
    # cdr3 = idx_from_ranges(cdr3)

    common_elements = []
    common_elems_cdr3 = []
    for element in interface:
        if element in cdrs:
            common_elements.append(element)
        if element in cdr3:
            common_elems_cdr3.append(element)

    return len(common_elements) / len(interface), len(common_elems_cdr3) / len(
        interface
    )
