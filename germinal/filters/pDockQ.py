"""
pDockQ: Protein-protein docking quality metrics and utilities.

This module provides functions for calculating pDockQ and related metrics
for evaluating the quality of predicted protein-protein interfaces,
notably using AlphaFold2-derived structures and confidence scores.

Attribution:
Bryant, P., Pozzati, G. & Elofsson, A. Improved prediction of protein-protein interactions using AlphaFold2.
Nat Commun 13, 1265 (2022). https://doi.org/10.1038/s41467-022-28865-w

"""

import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import mdtraj as md
from scipy.optimize import curve_fit
from Bio.PDB.PDBParser import PDBParser
from typing import Dict, List, Tuple
from scipy.spatial.distance import pdist, squareform


def parse_atm_record(line):
    """Get the atm record"""
    record = defaultdict()
    record["name"] = line[0:6].strip()
    record["atm_no"] = int(line[6:11])
    record["atm_name"] = line[12:16].strip()
    record["atm_alt"] = line[17]
    record["res_name"] = line[17:20].strip()
    record["chain"] = line[21]
    record["res_no"] = int(line[22:26])
    record["insert"] = line[26].strip()
    record["resid"] = line[22:29]
    record["x"] = float(line[30:38])
    record["y"] = float(line[38:46])
    record["z"] = float(line[46:54])
    record["occ"] = float(line[54:60])
    record["B"] = float(line[60:66])

    return record


def pdb_2_coords(pdb):
    """Read a pdb file predicted with AF and rewritten to conatin all chains"""
    chain_coords = defaultdict(list)
    plddt_dict = OrderedDict()

    for line in pdb.split("\n"):
        if not line.startswith("ATOM"):
            continue
        record = parse_atm_record(line)

        # Get CB - CA for GLY
        if record["atm_name"] == "CB" or (
            record["atm_name"] == "CA" and record["res_name"] == "GLY"
        ):
            chain_coords[record["chain"]].append(
                [record["x"], record["y"], record["z"]]
            )
            res_id = record["chain"] + str(record["res_no"])
            if res_id in plddt_dict.keys():
                plddt_dict[record["chain"] + str(record["res_no"])].append(record["B"])
            else:
                plddt_dict[record["chain"] + str(record["res_no"])] = [record["B"]]

        plddt = np.array([np.mean(plddts) for plddts in plddt_dict.values()])

    return chain_coords, plddt


def calc_pdockq(chain_coords, plddt, contact_cutoff=8.0):
    """
    N-chain pDockQ:
      - Builds a full distance matrix across all chains
      - Uses only cross-chain contacts (< contact_cutoff Å)
      - Averages plDDT over unique interface residues
      - Counts unique contacts (upper-triangular, cross-chain)
    Returns: (pdockq, avg_if_plddt, n_if_contacts)
    """
    # Concatenate coords in key order and track chain index per residue

    chains = list(chain_coords.keys())
    coords_list = []
    chain_idx = []
    for ci, ch in enumerate(chains):
        arr = np.asarray(chain_coords[ch], dtype=float)
        coords_list.append(arr)
        chain_idx.append(np.full(arr.shape[0], ci, dtype=int))
    coords = np.vstack(coords_list)
    chain_idx = np.concatenate(chain_idx)

    # Sanity check on lengths
    if coords.shape[0] != plddt.shape[0]:
        raise ValueError(
            f"Length mismatch: total residues from coords={coords.shape[0]} vs pLDDT={plddt.shape[0]}"
        )

    # Pairwise distances (N x N)
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))

    # Cross-chain mask (exclude same-chain and diagonal)
    same_chain = chain_idx[:, None] == chain_idx[None, :]
    cross_chain = ~same_chain
    np.fill_diagonal(cross_chain, False)

    # Contact mask: cross-chain & within cutoff
    contact_mask = (dists <= contact_cutoff) & cross_chain

    # Count unique contacts (upper triangle to avoid double counting)
    n_if_contacts = int(np.count_nonzero(np.triu(contact_mask, 1)))

    if n_if_contacts == 0:
        return 0.0, 0.0, 0

    # Interface residues = unique residues participating in any cross-chain contact
    sym_mask = contact_mask | contact_mask.T
    interface_res = np.where(sym_mask.any(axis=1))[0]

    # Average interface pLDDT
    avg_if_plddt = float(np.mean(plddt[interface_res]))

    # pDockQ formula
    x = avg_if_plddt * np.log10(n_if_contacts + 1)  # +1 to avoid log10(0)
    pdockq = 0.724 / (1.0 + np.exp(-0.052 * (x - 152.611))) + 0.018

    return float(pdockq), avg_if_plddt, n_if_contacts


def compute_pdockq(pdb):
    chain_coords, plddt = pdb_2_coords(pdb)
    pdockq, avg_if_plddt, n_if_contacts = calc_pdockq(chain_coords, plddt)
    return pdockq, avg_if_plddt, n_if_contacts, plddt.mean()


def get_pdockq(pdb_path):
    with open(pdb_path, "r+") as fh:
        pdb = fh.read()
        pdockq, _, _, _ = compute_pdockq(pdb)

    return pdockq


def retrieve_IFplddt(structure, chain1, chain2_lst, max_dist):
    ## generate a dict to save IF_res_id
    chain_lst = list(chain1) + chain2_lst

    ifplddt = []
    contact_chain_lst = []
    for res1 in structure[0][chain1]:
        for chain2 in chain2_lst:
            count = 0
            for res2 in structure[0][chain2]:
                if res1.has_id("CA") and res2.has_id("CA"):
                    dis = abs(res1["CA"] - res2["CA"])
                    ## add criteria to filter out disorder res
                    if dis <= max_dist:
                        ifplddt.append(res1["CA"].get_bfactor())
                        count += 1

                elif res1.has_id("CB") and res2.has_id("CB"):
                    dis = abs(res1["CB"] - res2["CB"])
                    if dis <= max_dist:
                        ifplddt.append(res1["CB"].get_bfactor())
                        count += 1
            if count > 0:
                contact_chain_lst.append(chain2)
    contact_chain_lst = sorted(list(set(contact_chain_lst)))

    if len(ifplddt) > 0:
        IF_plddt_avg = np.mean(ifplddt)
    else:
        IF_plddt_avg = 0

    return IF_plddt_avg, contact_chain_lst


def retrieve_IFPAEinter(structure, paeMat, contact_lst, max_dist):
    ## contact_lst:the chain list that have an interface with each chain. For eg, a tetramer with A,B,C,D chains and A/B A/C B/D C/D interfaces,
    ##             contact_lst would be [['B','C'],['A','D'],['A','D'],['B','C']]

    chain_lst = [x.id for x in structure[0]]
    seqlen = [len(x) for x in structure[0]]
    ifch1_col = []
    ifch2_col = []
    ch1_lst = []
    ch2_lst = []
    ifpae_avg = []
    d = 10
    for ch1_idx in range(len(chain_lst)):
        ## extract x axis range from the PAE matrix
        idx = chain_lst.index(chain_lst[ch1_idx])
        ch1_sta = sum(seqlen[:idx])
        ch1_end = ch1_sta + seqlen[idx]
        ifpae_col = []
        ## for each chain that shares an interface with chain1, retrieve the PAE matrix for the specific part.
        for contact_ch in contact_lst[ch1_idx]:
            index = chain_lst.index(contact_ch)
            ch_sta = sum(seqlen[:index])
            ch_end = ch_sta + seqlen[index]
            remain_paeMatrix = paeMat[ch1_sta:ch1_end, ch_sta:ch_end]

            ## get avg PAE values for the interfaces for chain 1
            mat_x = -1
            for res1 in structure[0][chain_lst[ch1_idx]]:
                mat_x += 1
                mat_y = -1
                for res2 in structure[0][contact_ch]:
                    mat_y += 1
                    if res1["CA"] - res2["CA"] <= max_dist:
                        ifpae_col.append(remain_paeMatrix[mat_x, mat_y])
        ## normalize by d(10A) first and then get the average
        if not ifpae_col:
            ifpae_avg.append(0)
        else:
            norm_if_interpae = np.mean(1 / (1 + (np.array(ifpae_col) / d) ** 2))
            ifpae_avg.append(norm_if_interpae)

    return ifpae_avg


def calc_pmidockq(ifpae_norm, ifplddt):
    df = pd.DataFrame()
    df["ifpae_norm"] = ifpae_norm
    df["ifplddt"] = ifplddt
    df["prot"] = df.ifpae_norm * df.ifplddt
    fitpopt = [
        1.31034849e00,
        8.47326239e01,
        7.47157696e-02,
        5.01886443e-03,
    ]  ## from orignal fit function
    df["pmidockq"] = sigmoid(df.prot.values, *fitpopt)

    return df


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k * (x - x0))) + b
    return y


def fit_newscore(df, column):
    testdf = df[df[column] > 0]

    colval = testdf[column].values
    dockq = testdf.DockQ.values
    xdata = colval[np.argsort(colval)]
    ydata = dockq[np.argsort(dockq)]

    p0 = [
        max(ydata),
        np.median(xdata),
        1,
        min(ydata),
    ]  # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0)  # method='dogbox', maxfev=50000)

    return popt


def pDockQ2(pdb_path, pae, distance=10.0):
    pdbp = PDBParser(QUIET=True)

    structure = pdbp.get_structure("", pdb_path)
    chains = []
    for chain in structure[0]:
        chains.append(chain.id)

    remain_contact_lst = []
    ## retrieve interface plDDT at chain-level
    plddt_lst = []
    for idx in range(len(chains)):
        chain2_lst = list(set(chains) - set(chains[idx]))
        IF_plddt, contact_lst = retrieve_IFplddt(
            structure, chains[idx], chain2_lst, distance
        )
        plddt_lst.append(IF_plddt)
        remain_contact_lst.append(contact_lst)

    avgif_pae = retrieve_IFPAEinter(structure, pae, remain_contact_lst, distance)
    ## calculate pmiDockQ

    res = calc_pmidockq(avgif_pae, plddt_lst)
    pdock_scores = np.array(res.get("pmidockq"))
    chains_result = {''.join(k): (v1, v2, v3) for k, v1, v2, v3 in zip(remain_contact_lst, plddt_lst, avgif_pae, pdock_scores)}
    
    return res, chains_result


def calculate_lis(
    pdb_path: str,
    pae_matrix: np.ndarray,
    pae_cutoff: float = 12,
    distance_cutoff: float = 8,
) -> Dict[str, np.ndarray]:
    """
    Calculate Local Interaction Score (LIS) for protein complexes.

    Args:
        pdb_path: Path to PDB file
        pae_matrix: PAE matrix (numpy array)
        pae_cutoff: PAE threshold for LIS calculation (default: 12)
        distance_cutoff: Distance threshold for contacts (default: 8Å)

    Returns:
        dict: Contains LIS, cLIS, iLIS, LIA, cLIA, LIR, cLIR scores
    """
    # Transform PAE matrix to LIS
    transformed_pae = _transform_pae_matrix(pae_matrix, pae_cutoff)

    # Get chain boundaries
    chain_lengths = _get_chain_lengths(pdb_path)
    subunit_sizes = list(chain_lengths.values())

    # Calculate contact map
    contact_map = _calculate_contact_map(pdb_path, distance_cutoff)

    # Calculate LIS matrix
    mean_lis_matrix = _calculate_mean_lis(transformed_pae, subunit_sizes)

    # Calculate cLIS (contact-based LIS)
    combined_map = np.where(
        (transformed_pae > 0) & (contact_map == 1), transformed_pae, 0
    )
    mean_clis_matrix = _calculate_mean_lis(combined_map, subunit_sizes)

    # Calculate iLIS = sqrt(LIS * cLIS)
    ilis_matrix = np.sqrt(mean_lis_matrix * mean_clis_matrix)

    # Calculate count-based metrics
    lia_matrix, lir_matrix, clia_matrix, clir_matrix = _calculate_count_metrics(
        transformed_pae, combined_map, subunit_sizes
    )

    return {
        "LIS": mean_lis_matrix,
        "cLIS": mean_clis_matrix,
        "iLIS": ilis_matrix,
        "LIA": lia_matrix,
        "cLIA": clia_matrix,
        "LIR": lir_matrix,
        "cLIR": clir_matrix,
    }


def _transform_pae_matrix(pae_matrix: np.ndarray, pae_cutoff: float) -> np.ndarray:
    """
    Transform PAE matrix to LIS scores.

    Args:
        pae_matrix: Input PAE matrix
        pae_cutoff: PAE threshold for transformation

    Returns:
        Transformed PAE matrix
    """
    transformed_pae = np.zeros_like(pae_matrix)
    within_cutoff = pae_matrix < pae_cutoff
    transformed_pae[within_cutoff] = 1 - (pae_matrix[within_cutoff] / pae_cutoff)
    return transformed_pae


def _get_chain_lengths(pdb_path: str) -> Dict[str, int]:
    """
    Get chain lengths from PDB file.

    Args:
        pdb_path: Path to PDB file

    Returns:
        Dictionary mapping chain IDs to their lengths
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    chain_lengths: Dict[str, int] = {}

    for chain in structure[0]:
        chain_lengths[chain.id] = len(chain)

    return chain_lengths


def _calculate_contact_map(pdb_path: str, distance_threshold: float = 8) -> np.ndarray:
    """
    Calculate contact map from PDB coordinates.

    Args:
        pdb_path: Path to PDB file
        distance_threshold: Distance threshold for contacts

    Returns:
        Contact map as numpy array
    """
    traj = md.load_pdb(pdb_path)

    # Select C-beta atoms (and C-alpha for Glycine)
    cb_selection = traj.topology.select("name CB or (resname GLY and name CA)")

    if cb_selection.size == 0:
        return np.array([])

    # Get coordinates of selected atoms
    coords = traj.xyz[0, cb_selection, :]

    # Calculate pairwise distances and contact map
    distances = squareform(pdist(coords))
    contact_map = (distances < distance_threshold).astype(int)

    return contact_map


def _calculate_mean_lis(
    transformed_pae: np.ndarray, subunit_sizes: List[int]
) -> np.ndarray:
    """
    Calculate mean LIS for each subunit pair.

    Args:
        transformed_pae: The PAE matrix after LIS transformation.
        subunit_sizes: A list of integer lengths for each chain.

    Returns:
        A 2D numpy array containing the mean LIS for each chain pair.
    """
    cum_lengths = np.cumsum(subunit_sizes)
    start_indices = np.concatenate(([0], cum_lengths[:-1]))

    mean_lis_matrix = np.zeros((len(subunit_sizes), len(subunit_sizes)))

    for i in range(len(subunit_sizes)):
        for j in range(len(subunit_sizes)):
            start_i, end_i = start_indices[i], cum_lengths[i]
            start_j, end_j = start_indices[j], cum_lengths[j]

            submatrix = transformed_pae[start_i:end_i, start_j:end_j]
            mean_lis = submatrix[submatrix > 0].mean()
            mean_lis_matrix[i, j] = mean_lis if not np.isnan(mean_lis) else 0

    return mean_lis_matrix


def _calculate_count_metrics(
    transformed_pae: np.ndarray, combined_map: np.ndarray, subunit_sizes: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate count-based metrics (LIA, LIR, cLIA, cLIR).

    Args:
        transformed_pae: The PAE matrix after LIS transformation.
        combined_map: The contact-filtered LIS map (for cLIA/cLIR).
        subunit_sizes: A list of integer lengths for each chain.

    Returns:
        A tuple of four 2D numpy arrays: (lia_matrix, lir_matrix, clia_matrix, clir_matrix).
    """
    n_subunits = len(subunit_sizes)
    lia_matrix = np.zeros((n_subunits, n_subunits), dtype=int)
    lir_matrix = np.zeros((n_subunits, n_subunits), dtype=int)
    clia_matrix = np.zeros((n_subunits, n_subunits), dtype=int)
    clir_matrix = np.zeros((n_subunits, n_subunits), dtype=int)

    cum_lengths = np.cumsum(subunit_sizes)
    starts = np.concatenate(([0], cum_lengths[:-1]))

    for i in range(n_subunits):
        for j in range(n_subunits):
            start_i, end_i = starts[i], cum_lengths[i]
            start_j, end_j = starts[j], cum_lengths[j]

            # LIA and LIR
            lia_submatrix = (transformed_pae[start_i:end_i, start_j:end_j] > 0).astype(
                int
            )
            lia_matrix[i, j] = np.count_nonzero(lia_submatrix)
            residues_i = np.unique(np.where(lia_submatrix > 0)[0]) + 1
            residues_j = np.unique(np.where(lia_submatrix > 0)[1]) + 1
            lir_matrix[i, j] = len(residues_i) + len(residues_j)

            # cLIA and cLIR
            clia_submatrix = (combined_map[start_i:end_i, start_j:end_j] > 0).astype(
                int
            )
            clia_matrix[i, j] = np.count_nonzero(clia_submatrix)
            residues_i = np.unique(np.where(clia_submatrix > 0)[0]) + 1
            residues_j = np.unique(np.where(clia_submatrix > 0)[1]) + 1
            clir_matrix[i, j] = len(residues_i) + len(residues_j)

    return lia_matrix, lir_matrix, clia_matrix, clir_matrix
