"""Filter utilities for structure prediction and interface quality assessment.

This module contains functions for running filters and computing metrics for a single design trajectory.
"""

import os
from tempfile import gettempdir
from typing import Any, Dict, Tuple, Sequence, Set, Union, List
import numpy as np
from iglm import IgLM
from germinal.utils import utils
from germinal.filters import af3, chai, pDockQ, pyrosetta_utils
from germinal.utils.io import IO, Trajectory


def run_filters(
    trajectory: Trajectory,
    run_settings: dict,
    target_settings: dict,
    filter_set: dict,
    io: IO,
    trajectory_sequence: str,
    trajectory_pdb_af: str,
    target_len: int,
) -> Tuple[dict, dict, bool, str]:
    """Run filters and compute metrics for a single design trajectory.

    The pipeline:
    1) Predict complex structure using AF3 or Chai on the final sequence
    2) Relax the predicted structure
    3) Compute clashes, secondary structure content, and interface metrics
    4) Compute additional confidence metrics (pDockQ, pDockQ2, LIS/LIA)
    5) Compute hydrophobic patch and hotspot proximity metrics
    6) Aggregate all metrics and evaluate against the provided filter set

    Args:
        trajectory: Trajectory metadata for the current design.
        run_settings: Configuration for the run (model choice, CDRs, etc.).
        target_settings: Target/binder chains, hotspots, target length.
        filter_set: Mapping of metric name to threshold spec with an operator.
        io: IO/layout helper providing directory structure.
        trajectory_sequence: Final amino acid sequence of the binder.
        trajectory_pdb_af: Path to the multimer PDB file used for alignment.

    Returns:
        Tuple containing:
            - filter_metrics: Dict of aggregated metrics for the trajectory
            - filter_results: Dict mapping '<metric>_filter' to pass/fail booleans
            - accepted: True if all filters in the set passed, else False
            - external_relaxed_pdb: Path to the relaxed complex PDB file
    """
    # ========================== Run Chai-1 or AF3 with final sequence ==========================
    structures_directory = io.layout.trajectories / "structures"
    target_chain = target_settings["target_chain"]
    binder_chain = target_settings["binder_chain"]
    target_sequence = []
    sequences_from_pdb = utils.get_sequence_from_pdb(run_settings["starting_pdb_complex"])

    for ch in target_chain.split(","):    
        target_sequence.append(sequences_from_pdb[
            ch
        ])
    if run_settings["type"].lower() == "nb":
        cdr3 = (
            np.array(
                run_settings["cdr_positions"][sum(run_settings["cdr_lengths"][:-1]) :]
            )
            + 1
        )
    elif run_settings["type"].lower() == "scfv":
        cdr3 = (
            np.array(
                run_settings["cdr_positions"][
                    sum(run_settings["cdr_lengths"][:2]) : sum(
                        run_settings["cdr_lengths"][:3]
                    )
                ]
            )
            + 1
        )
    else:
        raise ValueError(
            f"Type {run_settings['type']} not supported, select either nb or scfv"
        )

    external_pdb, external_metrics, ipsae = run_structure_prediction(
        trajectory_sequence=trajectory_sequence,
        target_sequence=target_sequence,
        target_chain=target_chain,
        binder_chain=binder_chain,
        structures_directory=structures_directory,
        design_name=trajectory.design_name,
        run_settings=run_settings,
        hotspot_residue = target_settings.get("hotspot_residue", None),
        target_len=target_len,
    )

    # ========================== FastRelax ==========================
    external_relaxed_pdb = os.path.join(
        structures_directory, trajectory.design_name + "_relaxed.pdb"
    )
    pyrosetta_utils.pr_relax(external_pdb, external_relaxed_pdb)

    # ========================== Calculate Clashes ==========================
    clash_threshold = run_settings["clash_threshold"]
    num_clashes_trajectory = utils.calculate_clash_score(
        external_pdb, threshold=clash_threshold
    )
    num_clashes_relaxed = utils.calculate_clash_score(
        external_relaxed_pdb, threshold=clash_threshold
    )

    # ========================== Secondary structure content ==========================
    ss_content = utils.calc_ss_percentage(
        external_pdb, run_settings, binder_chain, return_dict=True, target_chain=target_chain
    )

    # ========================== Calculate Interface Metrics ==========================
    interface_metric_names = ["interface_scores", "interface_AA", "interface_residues"]
    interface_metrics = {
        k: v
        for k, v in zip(
            interface_metric_names,
            pyrosetta_utils.score_interface(
                external_relaxed_pdb, binder_chain, target_chain=target_chain
            ),
        )
    }

    # ========================== Calculate number of framework mutations ==========================
    n_framework_mutations, framework_mutations = get_framework_mutations(
        trajectory_sequence,
        run_settings["starting_binder_seq"],
        run_settings["cdr_positions"],
    )
    print("Framework mutations:", framework_mutations)

    # ========================== Calculate Binding Interface (CDR 3) near hotspot filter ==========================
    one_indexed_cdr_positions = np.array(run_settings["cdr_positions"]) + 1

    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = (
        compute_hotspot_proximity(
            external_relaxed_pdb=external_relaxed_pdb,
            target_settings=target_settings,
            binder_chain=binder_chain,
            target_chain=target_chain,
            one_indexed_cdr_positions=one_indexed_cdr_positions,
            cdr3=cdr3,
            distance_threshold=run_settings["hotspot_distance_threshold"],
            contact_distance=run_settings["residue_contact_distance"],
            min_hotspot_contacts=run_settings["min_cdr_hotspot_contacts"],
        )
    )

    # ========================== Calculate Interface CDR % ==========================
    percent_interface_is_cdr = utils.interface_cdrs(
        interface_metrics["interface_residues"],
        run_settings["cdr_positions"],
        run_settings["cdr_positions"][sum(run_settings["cdr_lengths"][:-1]) :],
        binder_chain=binder_chain,
    )

    # ========================== Calculate pDockQ, pDockQ2, LIS/LIA ==========================
    pdockq_metrics, lis_metrics, pDockQ2_out = compute_pdockq_and_lis(
        external_pdb=external_pdb,
        external_metrics=external_metrics,
        binder_chain=binder_chain,
    )

    # ========================== Aggregate Confidence Metrics ==========================
    confidence_metrics = {
        "plddt": external_metrics["plddt"].item(),
        "plddt_binder": external_metrics["plddt_binder"].item(),
        "ptm": external_metrics["ptm"][0],
        "i_ptm": external_metrics["iptm"][0],
        "chain_ptm": external_metrics["chain_ptm"][-1],
        "pae": external_metrics["pae"].item(),
        "aggregate_score": external_metrics["aggregate_score"][0],
        "i_pae": pDockQ2_out["ifpae_norm"].mean(),
        "i_plddt": (pDockQ2_out["ifplddt"].mean() / 100),
        "binder_pae": external_metrics["binder_pae"].item(),
        "ipsae":ipsae
    }

    # ========================== Calculate Hydrophobic Patch Filter ==========================
    sap_score, cdr_sap, _, hydrophobic_patches_binder = pyrosetta_utils.get_sap_score(
        external_relaxed_pdb,
        binder_chain=binder_chain,
        only_binder=True,
        limit_sasa=run_settings["sap_limit_sasa"],
        patch_radius=run_settings["sap_patch_radius"],
        avg_sasa_patch_thr=run_settings["sap_avg_sasa_patch_thr"],
        cdrs=run_settings["cdr_positions"],
    )

    hydrophobic_patches_struct = []

    # ========================== Calculate RMSD of Binder between Multimer and External Predictor ==========================
    try:
        pyrosetta_utils.align_pdbs(
            external_pdb, trajectory_pdb_af, target_chain, target_chain
        )
        binder_rmsd = pyrosetta_utils.unaligned_rmsd(
            external_pdb, trajectory_pdb_af, binder_chain, binder_chain
        )

    except Exception:
        binder_rmsd = 100

    # ========================== Get Log-likelihood from IgLM ==========================
    if run_settings["ablm_model"] == "iglm":
        iglm_ll = get_iglm_ll(
            sequence=trajectory_sequence,
            species_token=run_settings["iglm_species"],
            vh_first=run_settings["vh_first"],
            vh_len=run_settings["vh_len"],
            vl_len=run_settings["vl_len"],
        )
    else:
        iglm_ll = -1
        print(f"Warning: {run_settings['ablm_model']} is not IgLM, skipping log-likelihood calculation")

    # ========================== Aggregate Filter Metrics ==========================
    filter_metrics = build_filter_metrics(
        confidence_metrics,
        interface_metrics,
        hydrophobic_patches_binder,
        hydrophobic_patches_struct,
        sap_score,
        cdr_sap,
        cdr3_hotspot_contacts,
        cdr_hotspot_contacts,
        pdockq_metrics,
        lis_metrics,
        percent_interface_is_cdr,
        ss_content,
        binder_rmsd,
        n_framework_mutations,
        framework_mutations,
        num_clashes_trajectory,
        num_clashes_relaxed,
        binder_near_hotspot,
        iglm_ll,
    )

    # ========================== Evaluate Filter Set ==========================
    accepted, filter_results = evaluate_filters(filter_set, filter_metrics)

    return filter_metrics, filter_results, accepted, external_relaxed_pdb


def build_filter_metrics(
    confidence_metrics: dict,
    interface_metrics: dict,
    hydrophobic_patches_binder,
    hydrophobic_patches_struct,
    sap_score,
    cdr_sap,
    cdr3_hotspot_contacts,
    cdr_hotspot_contacts,
    pdockq_metrics: dict,
    lis_metrics: dict,
    percent_interface_is_cdr,
    ss_content: dict,
    binder_rmsd,
    n_framework_mutations,
    framework_mutations,
    num_clashes_trajectory,
    num_clashes_relaxed,
    binder_near_hotspot,
    iglm_ll,
) -> Dict[str, Any]:
    """
    Aggregate all metrics into comprehensive evaluation dict (floats rounded to 4 decimals).

    Returns:
        Dict[str, Any]: Confidence, interface, structural, biological, and sequence metrics
    """
    ipsae = confidence_metrics["ipsae"]
    metrics = {
        # confidence
        "external_plddt": confidence_metrics["plddt"],
        "external_ptm": confidence_metrics["ptm"],
        "external_iptm": confidence_metrics["i_ptm"],
        "external_pae": confidence_metrics["pae"],
        "external_aggregate_score": confidence_metrics["aggregate_score"],
        "external_i_pae": confidence_metrics["i_pae"],
        "external_i_plddt": confidence_metrics["i_plddt"],
        "external_plddt_binder": confidence_metrics["plddt_binder"],
        "external_chain_ptm": confidence_metrics["chain_ptm"],
        "external_binder_pae": confidence_metrics["binder_pae"],
        "ipsae": None if ipsae is None else ipsae["ipsae"],
        # structure + interface
        "binder_near_hotspot": binder_near_hotspot,
        "clashes_unrelaxed": num_clashes_trajectory,
        "clashes": num_clashes_relaxed,  # relaxed clashes
        "binder_score": interface_metrics["interface_scores"]["binder_score"],
        "surface_hydrophobicity": interface_metrics["interface_scores"][
            "surface_hydrophobicity"
        ],
        "interface_shape_comp": interface_metrics["interface_scores"]["interface_sc"],
        "interface_packstat": interface_metrics["interface_scores"][
            "interface_packstat"
        ],
        "interface_dG": interface_metrics["interface_scores"]["interface_dG"],
        "interface_dSASA": interface_metrics["interface_scores"]["interface_dSASA"],
        "interface_dG_SASA_ratio": interface_metrics["interface_scores"][
            "interface_dG_SASA_ratio"
        ],
        "interface_fraction": interface_metrics["interface_scores"][
            "interface_fraction"
        ],
        "interface_hydrophobicity": interface_metrics["interface_scores"][
            "interface_hydrophobicity"
        ],
        "interface_nres": interface_metrics["interface_scores"]["interface_nres"],
        "interface_hbonds": interface_metrics["interface_scores"][
            "interface_interface_hbonds"
        ],
        "interface_hbond_percentage": interface_metrics["interface_scores"][
            "interface_hbond_percentage"
        ],
        "interface_delta_unsat_hbonds": interface_metrics["interface_scores"][
            "interface_delta_unsat_hbonds"
        ],
        "interface_delta_unsat_hbonds_percentage": interface_metrics[
            "interface_scores"
        ]["interface_delta_unsat_hbonds_percentage"],
        "hydrophobic_patches_binder": len(hydrophobic_patches_binder),
        "hydrophobic_patches_struct": len(hydrophobic_patches_struct),
        "sap_score": sap_score,
        "cdr_sap": cdr_sap,
        "cdr3_hotspot_contacts": cdr3_hotspot_contacts,
        "cdr_hotspot_contacts": cdr_hotspot_contacts,
        "binder_near_hotspot": binder_near_hotspot,
        # derived confidence
        "pdockq2": pdockq_metrics["pDockQ2"],
        "ipsae_pdockq2": None if ipsae is None else ipsae["pdockq2"],
        "lis_lis": lis_metrics["lis"],
        "lis_lia": lis_metrics["lia"],
        # secondary structure + framework metrics
        "percent_interface_cdr": percent_interface_is_cdr[0],
        "percent_interface_cdr3": percent_interface_is_cdr[1],
        "alpha_interface": ss_content["alpha_i"],
        "beta_interface": ss_content["beta_i"],
        "loops_interface": ss_content["loops_i"],
        "alpha_all": ss_content["alpha_"],
        "beta_all": ss_content["beta_"],
        "loops_all": ss_content["loops_"],
        "sc_rmsd": binder_rmsd,
        "n_framework_mutations": n_framework_mutations,
        # large logs
        "framework_mutations": framework_mutations,
        "interface_AA": interface_metrics["interface_AA"],
        "interface_residues": interface_metrics["interface_residues"],
        "ss_content": ss_content,
        # iglm
        "iglm_ll": iglm_ll,
    }

    # round floats to 4 decimals for compactness
    metrics = {
        k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
    }
    return metrics


def evaluate_filters(
    filter_set: dict, filter_metrics: dict
) -> Tuple[bool, Dict[str, bool]]:
    """
    Evaluate metrics against quality filters (operators: <, <=, >, >=, ==, =).

    Args:
        filter_set: {metric_name: {"value": threshold, "operator": op}}
        filter_metrics: Calculated metrics dict

    Returns:
        Tuple[bool, Dict[str, bool]]: (all_passed, individual_results)
    """
    filter_results = {}

    for filter_name, filter_config in filter_set.items():
        if filter_name not in filter_metrics:
            print(f"Warning: Filter '{filter_name}' not found in metrics, skipping")
            filter_results[f"{filter_name}_filter"] = False
            continue

        metric_value = filter_metrics[filter_name]
        threshold = filter_config["value"]
        operator = filter_config["operator"]

        # Evaluate based on operator
        if operator == "<":
            passed = metric_value < threshold
        elif operator == "<=":
            passed = metric_value <= threshold
        elif operator == ">":
            passed = metric_value > threshold
        elif operator == ">=":
            passed = metric_value >= threshold
        elif operator == "==":
            passed = metric_value == threshold
        elif operator == "=":
            passed = metric_value == threshold
        else:
            print(f"Warning: Unknown operator '{operator}' for filter '{filter_name}'")
            passed = False

        filter_results[f"{filter_name}_filter"] = passed

    # All filters must pass
    all_passed = all(filter_results.values())

    return all_passed, filter_results


def get_framework_mutations(
    trajectory_sequence: str,
    framework_sequence: str,
    cdr_positions: Sequence[int],
) -> Tuple[int, List[str]]:
    """
    Identify mutations outside CDR regions (format: 'A123B').

    Args:
        trajectory_sequence: Final designed sequence
        framework_sequence: Original reference sequence
        cdr_positions: CDR positions (0-indexed)

    Returns:
        Tuple[int, List[str]]: (count, mutation_list)
    """
    framework_mutations = []
    for i, (seq, framework) in enumerate(zip(trajectory_sequence, framework_sequence)):
        if seq != framework and i not in cdr_positions:
            framework_mutations.append(f"{framework}{i + 1}{seq}")
    return len(framework_mutations), framework_mutations


def is_binder_near_hotspot(
    target_contacts: Sequence[int],
    target_hotspots: Sequence[int],
    binder_contacts: Sequence[int],
    cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    min_hotspot_contacts: int = 3,
) -> Union[Tuple[bool, int, int], Tuple[Set[int], Set[int]]]:
    """Check whether the binder interface is near target hotspot residues.

    Args:
        target_contacts: Target residue indices at the interface (1-indexed).
        target_hotspots: Target hotspot residue indices (1-indexed).
        binder_contacts: Binder residue indices at the interface (1-indexed).
        cdr_positions: Binder CDR residue indices (1-indexed).
        cdr3: Binder CDR3 residue indices (1-indexed).
        return_bool: If True, return summary booleans/counts; otherwise, return
            the sets of binder residues contacting hotspots for CDR3 and all CDRs.

    Returns:
        If return_bool is True:
            (has_min_cdr_contacts, num_cdr3_hotspot_contacts, num_cdr_hotspot_contacts)
        Else:
            (cdr3_hotspot_contact_set, cdr_hotspot_contact_set)
    """
    cdr_hotspot_contacts = []
    cdr3_hotspot_contacts = []
    for i, tc in enumerate(target_contacts):
        if tc in target_hotspots:
            if binder_contacts[i] in cdr_positions:
                cdr_hotspot_contacts.append(binder_contacts[i])
            if binder_contacts[i] in cdr3:
                cdr3_hotspot_contacts.append(binder_contacts[i])
    cdr_hotspot_contacts = set(cdr_hotspot_contacts)
    cdr3_hotspot_contacts = set(cdr3_hotspot_contacts)
    accept_critera = len(cdr_hotspot_contacts) >= min_hotspot_contacts

    return accept_critera, len(cdr3_hotspot_contacts), len(cdr_hotspot_contacts)


def run_structure_prediction(
    trajectory_sequence: str,
    target_sequence: str,
    target_chain: str,
    binder_chain: str,
    structures_directory,
    design_name: str,
    run_settings: dict,
    target_len: int,
    hotspot_residue = None,
) -> Tuple[str, dict]:
    """
    Run AF3 or Chai structure prediction for antibody-target complex.

    Args:
        trajectory_sequence: Designed antibody sequence
        target_sequence: Target protein sequence
        target_chain: Target chain ID
        binder_chain: Binder chain ID
        structures_directory: Output directory
        design_name: Design identifier
        run_settings: Config with model choice and parameters

    Returns:
        Tuple[str, dict]: (pdb_path, confidence_metrics)
    """
    af3_seed = [int(x) for x in np.random.randint(0, 999999, size=3)]
    ipsae = None
    if run_settings["structure_model"] == "af3":
        external_pdb, external_metrics, ipsae = af3.run_af3(
            trajectory_sequence,
            target_sequence,
            target_chain,
            structures_directory,
            design_name,
            af3_seed,
            run_settings,
            binder_chain=binder_chain,
            msa_mode=run_settings["msa_mode"],
        )
    elif run_settings["structure_model"] == "chai":

        cdr3_idx = run_settings["cdr_positions"][run_settings["cdr_lengths"][0] + run_settings["cdr_lengths"][1]:]
        cdr3_idx = cdr3_idx[len(cdr3_idx)//2]

        external_pdb, external_metrics = chai.run_chai(
            trajectory_sequence,
            gettempdir(),
            structures_directory,
            run_settings["starting_pdb_complex"],
            target_chain,
            seed=af3_seed[0],
            cdr3_idx = cdr3_idx,
            hotspot_residue = hotspot_residue,
            binder_chain=binder_chain,
            target_len=target_len,
        )
    else:
        raise ValueError(
            f"Structure model {run_settings['structure_model']} not supported, select either af3 or chai"
        )

    return external_pdb, external_metrics, ipsae


def compute_hotspot_proximity(
    external_relaxed_pdb: str,
    target_settings: dict,
    binder_chain: str,
    target_chain: str,
    one_indexed_cdr_positions: Sequence[int],
    cdr3: Sequence[int],
    distance_threshold: float = 5.3,
    contact_distance: float = 6.0,
    min_hotspot_contacts: int = 3,
) -> Tuple[bool, int, int]:
    """
    Compute CDR contacts with target hotspot residues (5.3Å threshold, ≥3 contacts required).

    Args:
        external_relaxed_pdb: Relaxed complex PDB path
        target_settings: Config with hotspot definitions
        binder_chain: Binder chain ID
        target_chain: Target chain ID
        one_indexed_cdr_positions: All CDR positions (1-indexed)
        cdr3: CDR3 positions (1-indexed)

    Returns:
        Tuple[bool, int, int]: (near_hotspot, cdr3_contacts, cdr_contacts)
    """
    # Default values when no hotspot specification is provided
    binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts = True, 0, 0
    offset = 0
    binder_near_hotspot = []
    cdr3_hotspot_contacts_ch = 0
    cdr_hotspot_contacts_ch = 0

    if len(target_settings["target_hotspots"]) > 0:
        target_chains = target_chain.split(",")
        for ch in target_chains:

            target_hotspots = np.array(utils.idx_from_ranges(target_settings["target_hotspots"],ch,offset=offset))+1

            hotspot_region = pyrosetta_utils.find_nearby_residues_from_pdb(
                external_relaxed_pdb,
                target_hotspots,
                distance_threshold=distance_threshold,
                chain=ch,
            )

            contacts = pyrosetta_utils.get_residue_contacts(
                external_relaxed_pdb, ch, binder_chain, contact_distance
            )
            contacts_per_chain = np.array(list(contacts.keys()))

            try:
                binder_near_chain_ht, cdr3_hotspot_contacts_ch, cdr_hotspot_contacts_ch = (
                    is_binder_near_hotspot(
                        contacts_per_chain[:, 0],
                        hotspot_region,
                        contacts_per_chain[:, 1],
                        one_indexed_cdr_positions,
                        cdr3,
                        min_hotspot_contacts=min_hotspot_contacts,
                    )
                )
            except Exception:
                binder_near_chain_ht, cdr3_hotspot_contacts_ch, cdr_hotspot_contacts_ch = (False, 0, 0) 

            binder_near_hotspot.append(binder_near_chain_ht)
            cdr3_hotspot_contacts += cdr3_hotspot_contacts_ch
            cdr_hotspot_contacts += cdr_hotspot_contacts_ch
        binder_near_hotspot = all(binder_near_hotspot)

    return binder_near_hotspot, cdr3_hotspot_contacts, cdr_hotspot_contacts


def compute_pdockq_and_lis(
    external_pdb: str,
    external_metrics: dict,
    binder_chain: str,
) -> Tuple[dict, dict, dict]:
    """
    Compute docking quality metrics: pDockQ, pDockQ2, LIS, LIA (0-1 scale, higher=better).

    Args:
        external_pdb: Complex PDB path
        external_metrics: Metrics with PAE matrix

    Returns:
        Tuple[dict, dict, dict]: (pdockq_metrics, lis_metrics, pDockQ2_out)
    """
    external_pae = external_metrics["pae_matrix"]
    pDockQ2_out, chain_specific_pdockq2 = pDockQ.pDockQ2(external_pdb, external_pae)
    pDockQ2 = []

    for i in chain_specific_pdockq2.keys():
        if binder_chain in i:
            pDockQ2.append(chain_specific_pdockq2[i][-1])

    pdockq_metrics = {
        "pDockQ2": np.mean(pDockQ2),
    }

    raw_lis_metrics = pDockQ.calculate_lis(external_pdb, external_pae)
    lis_metrics = {
        "lis": np.mean([raw_lis_metrics["LIS"][0, 1], raw_lis_metrics["LIS"][1, 0]]),
        "lia": np.mean([raw_lis_metrics["LIA"][0, 1], raw_lis_metrics["LIA"][1, 0]]),
    }

    return pdockq_metrics, lis_metrics, pDockQ2_out


def get_iglm_ll(
    sequence,
    chain_token="[HEAVY]",
    species_token="[CAMEL]",
    vh_first=True,
    vh_len=0,
    vl_len=0,
):
    """
    Calculate antibody sequence log-likelihood using IgLM language model.

    Attribution: Shuai, R. W., Ruffolo, J. A., & Gray, J. J. (2023). IgLM: Infilling
    language modeling for antibody sequence design. Cell Systems, 14(11), 979-989.
    License: JHU Academic Software License (non-commercial use). Commercial inquiries: awichma2@jhu.edu
    Source: https://github.com/Graylab/IgLM

    Args:
        sequence: Antibody amino acid sequence
        chain_token: "[HEAVY]" or "[LIGHT]"
        species_token: "[HUMAN]", "[CAMEL]", etc.
        vh_first: Heavy chain first in scFv sequence
        vh_len: Heavy chain length (0 for nanobodies)
        vl_len: Light chain length (0 for nanobodies)

    Returns:
        float: Log-likelihood score (higher = more natural)
    """

    # Initialize the model
    model = IgLM()

    # Compute the log likelihood, depending on nanobody or scfv
    if vl_len and vh_len:
        if vh_first:
            log_likelihood_h = model.log_likelihood(
                sequence[:vh_len], "[HEAVY]", species_token
            )
            log_likelihood_l = model.log_likelihood(
                sequence[-vl_len:], "[LIGHT]", species_token
            )
        else:
            log_likelihood_l = model.log_likelihood(
                sequence[:vl_len], "[LIGHT]", species_token
            )
            log_likelihood_h = model.log_likelihood(
                sequence[-vh_len:], "[HEAVY]", species_token
            )
        log_likelihood = log_likelihood_h + log_likelihood_l
    else:
        log_likelihood = model.log_likelihood(sequence, chain_token, species_token)

    return log_likelihood
