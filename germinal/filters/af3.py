"""
Run AlphaFold 3 for antibody structure prediction.

This script is based on the AlphaFold 3  from the AlphaFold 3 repository:
https://github.com/deepmind/alphafold/

Attribution:
If you use this code, the AlphaFold 3 model, or outputs produced by it in your research, please cite:

Abramson, J., Adler, J., Dunger, J., Evans, R., Green, T., Pritzel, A., Ronneberger, O., Willmore, L., Ballard, A. J., Bambrick, J., et al. (2024). Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 630(8016), 493–500. https://doi.org/10.1038/s41586-024-07487-w

Copyright 2024 DeepMind Technologies Limited.

License:
- The AlphaFold 3 source code is licensed under the Creative Commons Attribution-Non-Commercial ShareAlike International License, Version 4.0 (CC-BY-NC-SA 4.0). See: https://github.com/google-deepmind/alphafold3/blob/main/LICENSE
- The AlphaFold 3 model parameters are subject to the AlphaFold 3 Model Parameters Terms of Use: https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

You may not use this file or the model parameters except in compliance with these terms.
"""

import subprocess
import argparse
import tempfile
import os
import json
import shutil
import numpy as np
import pandas as pd

from typing import Union, List
from colabfold.colabfold import run_mmseqs2
from Bio import PDB
from concurrent.futures import ProcessPoolExecutor, TimeoutError


def create_input_dict(
    binder_seq, target_seq, binder_chain, target_chains, design_name, seed
):
    """
    Create input JSON data for AlphaFold3 inference.

    Args:
        binder_seq (str): Amino acid sequence of the binder protein.
        target_seq (str): Amino acid sequence of the target protein.
        binder_chain (str): Chain ID for the binder protein.
        target_chains (Union[str, List[str]]): Chain ID(s) for the target protein.
        design_name (str): Name identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for model inference.

    Returns:
        dict: AF3-compatible input JSON structure.
    """
    if isinstance(target_chains, str):
        target_chains = target_chains.split(',')
        
    if isinstance(seed, int):
        seed = [seed]

    input_json_data = {
        "name": design_name,
        "modelSeeds": seed,
        "dialect": "alphafold3",
        "version": 2,
    }

    sequences = []
    for chain_id in [binder_chain] + target_chains:
        if chain_id != binder_chain:
            i = target_chains.index(chain_id)
        sequences.append(
            {
                "protein": {
                    "id": [chain_id],
                    "sequence": binder_seq if chain_id == binder_chain else target_seq[i],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                }
            }
        )
    input_json_data["sequences"] = sequences
    return input_json_data


def remove_a3m_insertions(a3m_path):
    """
    Remove insertion characters from A3M MSA file for AF3 compatibility.

    AlphaFold3 requires MSA sequences to have uniform length, so we remove
    lowercase insertion characters that indicate gaps in the alignment.

    Args:
        a3m_path (str): Path to the A3M format MSA file to process.
    """
    with open(a3m_path, "r") as a3m_file:
        lines = a3m_file.readlines()
    new_lines = []
    for line in lines:
        line = line.replace("\x00", "")
        if line.startswith("#") or line.startswith(">"):
            new_lines.append(line)
        else:
            new_lines.append("".join(c for c in line if not c.islower()))
    with open(a3m_path, "w") as a3m_file:
        a3m_file.writelines(new_lines)


def generate_local_msa(
    sequence,
    design_name,
    output_dir,
    msa_db_dir,
    use_gpu=False,
    use_gpu_server=False,
    use_metagenomic_db=False,
):
    """
    Generate an unpaired MSA for the given sequence using colabfold_search.

    Args:
        sequence (str): The sequence to generate an MSA for.
        design_name (str): The name of the design. Used to name the output file.
        output_dir (str): The directory to save the output files to.
        msa_db_dir (str): The directory containing the MSA databases.
        use_gpu (bool): Whether to use a GPU for the search.
        use_gpu_server (bool): Whether to use a GPU server for the search.
        use_metagenomic_db (bool): Whether to use the metagenomic database.
    """
    # Start GPU server processes for accelerated MSA search
    if use_gpu_server:
        print("Starting GPU server...")
        gpu_server_dir = os.path.join(msa_db_dir, "colabfold_envdb_202108_db")
        uniref30_db_dir = os.path.join(msa_db_dir, "uniref30_2302_db")
        gpu_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                gpu_server_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("GPU server started at PID", gpu_server_process.pid)
        gpu_server_process.wait()
        uniref30_server_process = subprocess.Popen(
            [
                "mmseqs",
                "gpuserver",
                uniref30_db_dir,
                "--max-seqs",
                "10000",
                "--db-load-mode",
                "0",
                "--prefilter-mode",
                "1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Uniref30 server started at PID", uniref30_server_process.pid)
        uniref30_server_process.wait()

    # Create temporary FASTA file for sequence input
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, f"{design_name}.fasta")
        with open(fasta_path, "w") as fasta_file:
            fasta_file.write(f">{design_name}\n{sequence}\n")
        # Create output directory for MSA results
        msa_out_dir = os.path.join(output_dir, "msas")
        os.makedirs(msa_out_dir, exist_ok=True)
        # Build ColabFold search command with appropriate flags
        cmd = ["colabfold_search"]
        if use_gpu:
            cmd += ["--gpu", "1"]
        if use_gpu_server:
            cmd += ["--gpu-server", "1"]
        if not use_metagenomic_db:
            cmd += ["--use-env", "0"]
        cmd += [fasta_path, msa_db_dir, msa_out_dir]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA."
            )
            return ""
        # Locate generated A3M file and rename it appropriately
        a3m_file = os.path.join(msa_out_dir, f"0.a3m")
        if os.path.exists(a3m_file):
            # Rename MSA file to match design name
            shutil.move(a3m_file, os.path.join(msa_out_dir, f"{design_name}.a3m"))
            a3m_file = os.path.join(msa_out_dir, f"{design_name}.a3m")
            # Process MSA to ensure uniform sequence length for AF3
            remove_a3m_insertions(a3m_file)
            return os.path.relpath(a3m_file, output_dir)
        else:
            print(
                f"colabfold_search failed for {design_name}: MSA not found at {a3m_file}. Falling back to no MSA."
            )
            return ""


def call_generate_colabfold_msa_with_timeout(
    sequence, design_name, output_dir, timeout=120, use_metagenomic_db=False
):
    """
    Generate MSA via ColabFold API with timeout protection.

    Uses ProcessPoolExecutor to enforce a timeout on MSA generation.
    If timeout is exceeded, the worker process is terminated and an empty
    MSA path is returned to allow AF3 to continue without MSA.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output naming.
        output_dir (str): Directory to save MSA output.
        timeout (int): Maximum time in seconds to wait for MSA.
        use_metagenomic_db (bool): Whether to include metagenomic databases.

    Returns:
        str: Relative path to generated MSA file, or empty string if failed.
    """

    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(
            generate_colabfold_msa,
            sequence,
            design_name,
            output_dir,
            use_metagenomic_db,
        )
        try:
            return fut.result(timeout=timeout)
        except TimeoutError:
            # Cancel and force-shutdown the pool, killing the worker process.
            fut.cancel()
            exe.shutdown(wait=False, cancel_futures=True)
            print(
                f"colabfold_search failed for {design_name}: timed out after {timeout}s. Returning empty MSA."
            )
            return ""


def generate_colabfold_msa(sequence, design_name, output_dir, use_metagenomic_db=False):
    """
    Generate unpaired MSA using ColabFold's remote API.

    This function calls the ColabFold web service to generate a multiple
    sequence alignment for the input protein sequence.

    Args:
        sequence (str): Protein sequence for MSA generation.
        design_name (str): Design identifier for output file naming.
        output_dir (str): Directory to save MSA results.
        use_metagenomic_db (bool): Whether to search metagenomic databases.

    Returns:
        str: Relative path to generated MSA file, or empty string if failed.
    """
    try:
        print(
            f"Running colabfold_search for {design_name} with use_env={use_metagenomic_db}"
        )
        run_mmseqs2(
            sequence,
            os.path.join(output_dir, f"{design_name}"),
            use_env=use_metagenomic_db,
        )
        print(f"colabfold_search finished for {design_name}")
    except Exception as e:
        print(
            f"colabfold_search failed for {design_name}: {e}. Falling back to no MSA."
        )
        return ""

    old_msa_path = os.path.join(output_dir, f"{design_name}_all", "uniref.a3m")
    # Process MSA to ensure uniform sequence length for AF3 compatibility
    remove_a3m_insertions(old_msa_path)
    if os.path.exists(old_msa_path):
        new_msa_path = os.path.join(output_dir, f"msas/{design_name}.a3m")
        os.makedirs(os.path.dirname(new_msa_path), exist_ok=True)
        shutil.copyfile(old_msa_path, new_msa_path)
        shutil.rmtree(os.path.join(output_dir, f"{design_name}_all"))
        return os.path.relpath(new_msa_path, output_dir)
    else:
        print(
            f"colabfold_search failed for {design_name}: MSA not found at {old_msa_path}. Falling back to no MSA."
        )
        return ""


def generate_msas(
    input_json_data: dict,
    msa_db_dir: str,
    output_dir: str,
    binder_chain: str,
    msa_mode: str,
    use_metagenomic_db: bool = False,
) -> dict:
    """
    Generate Multiple Sequence Alignments (MSAs) for protein chains.

    Creates MSAs for each protein chain in the input data using the specified
    method. MSAs improve structure prediction accuracy by providing evolutionary
    context. The function supports different MSA generation modes:
    - "local": Use local ColabFold search with databases
    - "colabfold": Use ColabFold remote API
    - "target": Generate MSA only for target protein

    Args:
        input_json_data (dict): AF3 input JSON containing sequence information.
        msa_db_dir (str): Path to local MSA databases (for local mode).
        output_dir (str): Directory to save generated MSA files.
        binder_chain (str): Chain identifier for the binder protein.
        msa_mode (str): MSA generation method.
        use_metagenomic_db (bool): Include metagenomic databases in search.

    Returns:
        dict: Updated input JSON with MSA paths added to each sequence.
    """
    updated_sequences = []
    for seq_idx, sequence_info in enumerate(input_json_data["sequences"]):
        chain = sequence_info["protein"]["id"][0]
        sequence = sequence_info["protein"]["sequence"]
        if chain != binder_chain:
            # Check if target MSA already exists to avoid regeneration
            design_name = f"target_{chain}"
            relative_msa_path = os.path.join(f"msas/{design_name}.a3m")
            full_msa_path = os.path.join(output_dir, relative_msa_path)

            if os.path.exists(full_msa_path):
                sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path
                updated_sequences.append(sequence_info)
                continue
        else:
            design_name = input_json_data["name"]
            # Skip binder MSA generation when mode is target-only
            if msa_mode == "target":
                updated_sequences.append(sequence_info)
                # print(f"Skipping MSA generation for {design_name} because msa_mode is target")
                continue

        # Generate MSA using the specified method
        if msa_mode == "local":
            relative_msa_path = generate_local_msa(
                sequence,
                design_name,
                output_dir,
                msa_db_dir,
                use_metagenomic_db=use_metagenomic_db,
            )
        elif msa_mode == "colabfold" or msa_mode == "target":
            relative_msa_path = call_generate_colabfold_msa_with_timeout(
                sequence, design_name, output_dir, use_metagenomic_db=use_metagenomic_db
            )
        else:
            print(f"MSA mode {msa_mode} not recognized. Skipping MSA generation.")

        sequence_info["protein"]["unpairedMsaPath"] = relative_msa_path

        updated_sequences.append(sequence_info)
        if relative_msa_path != "":
            print(
                f"Generated MSA at {relative_msa_path} for {design_name} and sequence {seq_idx}"
            )
        else:
            print(f"No MSA generated for {design_name} and sequence {seq_idx}")
    input_json_data["sequences"] = updated_sequences
    return input_json_data


def extract_structure_and_scores(output_dir, design_name, binder_chain):
    """
    Extract predicted structure and confidence scores from AF3 output.

    Processes AF3 results by converting the output CIF file to PDB format,
    extracting confidence metrics from JSON files, and cleaning up temporary
    files to save disk space.

    Args:
        output_dir (str): Directory containing AF3 output folder.
        design_name (str): Name identifier for the design.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to converted PDB structure file
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
    """

    af3_results_folder = os.path.join(output_dir, design_name)
    # Convert mmCIF structure file to PDB format for compatibility
    af3_structure = os.path.join(af3_results_folder, f"{design_name}_model.cif")
    pdb_path = os.path.join(output_dir, f"{design_name}_af3.pdb")
    parser = PDB.MMCIFParser(QUIET=True)
    io = PDB.PDBIO()
    structure = parser.get_structure("structure", af3_structure)
    io.set_structure(structure)
    io.save(pdb_path)
    # Extract confidence scores from AF3 JSON output files
    summary_confidences = os.path.join(
        af3_results_folder, f"{design_name}_summary_confidences.json"
    )
    full_confidences = os.path.join(
        af3_results_folder, f"{design_name}_confidences.json"
    )
    af3_scores = {}
    with open(summary_confidences, "r") as f:
        summary_metrics = json.load(f)
    with open(full_confidences, "r") as f:
        full_metrics = json.load(f)
    #plddt
    binder_plddt = [x for x, c in zip(full_metrics["atom_plddts"], full_metrics["atom_chain_ids"]) if c == binder_chain]
    af3_scores["plddt_binder"] = np.mean(binder_plddt) / 100 if binder_plddt else 0
    af3_scores["plddt"] = np.mean(full_metrics["atom_plddts"]) / 100
    #pae
    pae_matrix = np.array(full_metrics["pae"])
    af3_scores["pae_matrix"] = pae_matrix
    af3_scores["pae"] = np.mean(pae_matrix)
    af3_scores["chain_pair_pae_min"] = np.array(summary_metrics['chain_pair_pae_min'])
    binder_mask = np.array(full_metrics['token_chain_ids']) == binder_chain
    pae_chain = pae_matrix[np.ix_(binder_mask, binder_mask)]  # double mask: rows AND columns
    af3_scores["binder_pae"] = np.mean(pae_chain)
    #ptm
    af3_scores["ptm"] = [summary_metrics["ptm"]]
    af3_scores["iptm"] = [summary_metrics["iptm"]]
    af3_scores["chain_iptm"] = np.array(summary_metrics["chain_iptm"])
    af3_scores['chain_ptm'] = np.array(summary_metrics["chain_ptm"])
    #agg score
    af3_scores["aggregate_score"] = [summary_metrics["ranking_score"]]
    
    try:
        ipsae_result = subprocess.run(
            ["ipsae", full_confidences, af3_structure, "10", "10"],
            capture_output=True,
            text=True
        )
        ipsae_out_file = f"{os.path.dirname(os.path.abspath(af3_structure))}/{os.path.basename(af3_structure).replace('.cif','_10_10.txt')}"
        df = pd.read_csv(ipsae_out_file, delim_whitespace=True)
        ipsae_score = df['ipSAE'][len(df['ipSAE'])-1]
        pdockq2_score = df['pDockQ2'][len(df['pDockQ2'])-1]
        lis_score = df['LIS'][len(df['LIS'])-1]
        ipsae = {'ipsae':ipsae_score, 'pdockq2':pdockq2_score, 'LIS':lis_score}
    except:
        print(f'ipSAE calculation failed!')
        ipsae = None

    # Clean up temporary AF3 job folder to save disk space
    shutil.rmtree(af3_results_folder)

    return pdb_path, af3_scores, ipsae


def _run_af3(
    input_json: dict,
    output_dir: str,
    binder_chain: str,
    msa_mode: str,
    run_settings: dict,
) -> tuple:
    """
    Execute AlphaFold3 structure prediction via Singularity container.

    Runs AF3 inference using the provided input JSON and configuration settings.
    The function handles MSA generation, container execution, and result extraction.

    Args:
        input_json (dict): AF3-compatible input JSON with sequence information.
        output_dir (str): Directory to save prediction outputs.
        binder_chain (str): Chain identifier for the binder protein.
        msa_mode (str): MSA generation mode ("none", "local", "colabfold", "target").
        run_settings (dict): Configuration containing all AF3 paths and settings.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to predicted structure in PDB format
            - scores_dict (dict): Confidence metrics and scores
    """
    # Verify output directory exists
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print(f"Directory created at {output_dir}.")

    # Process input. if a path, load the dict, and get the dir name. otherwise, use a temp dir.
    input_dir = os.path.join(output_dir, "af3_inputs")
    os.makedirs(input_dir, exist_ok=True)
    input_path = os.path.join(input_dir, f"{input_json['name']}.json")
    # Generate MSAs if requested. pass input_dir as the output_dir to save msas there
    if msa_mode in ["local", "colabfold", "target"]:
        input_json = generate_msas(
            input_json,
            run_settings["msa_db_dir"],
            input_dir,
            binder_chain,
            msa_mode,
            use_metagenomic_db=run_settings["use_metagenomic_db"],
        )

    # Write updated JSON for AF3
    with open(input_path, "w") as f:
        json.dump(input_json, f)

    af3_repo_path = run_settings["af3_repo_path"]
    weights_path = run_settings["af3_model_dir"]
    databases_path = run_settings["af3_db_dir"]
    sif_path = run_settings["af3_sif_path"]

    run_cmds = [
        "singularity",
        "exec",
        "--nv",
        "--env",
        "LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu",
        "--bind",
        f"{output_dir}:/root/af_output",
        "--bind",
        f"{input_dir}:/root/af_input",
        "--bind",
        f"{weights_path}:/root/models",
        "--bind",
        f"{databases_path}:/root/public_databases",
        "--bind",
        f"{af3_repo_path}:/root/alphafold3",
        sif_path,
        "python",
        "/root/alphafold3/run_alphafold.py",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        f"--output_dir={output_dir}",
        f"--json_path={input_path}",
    ]

    popen = subprocess.Popen(
        run_cmds,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    ) 
    for line in popen.stdout:
        continue

    popen.stdout.close()
    return_code = popen.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, run_cmds)

    pdb_path, scores, ipsae = extract_structure_and_scores(output_dir, input_json["name"], binder_chain)

    return pdb_path, scores, ipsae


def run_af3(
    binder_seq: str,
    target_seq: str,
    target_chains: Union[List[str], str],
    output_dir: str,
    design_name: str,
    seed: Union[int, List[int]],
    run_settings: dict,
    binder_chain: str = "B",
    msa_mode: str = "none",
):
    """
    Run AlphaFold3 structure prediction for antibody-target complex.

    High-level interface for AF3 structure prediction. Creates input JSON,
    generates MSAs if requested, runs AF3 inference, and returns the predicted
    structure with confidence scores.

    Args:
        binder_seq (str): Amino acid sequence of the antibody binder.
        target_seq (str): Amino acid sequence of the target protein.
        target_chains (Union[List[str], str]): Chain identifier(s) for target protein.
        output_dir (str): Directory to save prediction outputs.
        design_name (str): Unique identifier for this design.
        seed (Union[int, List[int]]): Random seed(s) for reproducible predictions.
        run_settings (dict): Configuration containing AF3 paths and MSA settings:
            - af3_repo_path: Path to AlphaFold3 repository
            - af3_sif_path: Path to AF3 Singularity image
            - af3_model_dir: Path to AF3 model weights
            - af3_db_dir: Path to AF3 public databases
            - msa_db_dir: Path to ColabFold MSA databases
            - use_metagenomic_db: Whether to use metagenomic databases
        binder_chain (str, optional): Chain ID for binder protein. Defaults to 'B'.
        msa_mode (str, optional): MSA generation method:
            - 'none': No MSA generation
            - 'local': Use local ColabFold databases
            - 'colabfold': Use ColabFold remote API
            - 'target': Generate MSA only for target protein
            Defaults to 'none'.

    Returns:
        tuple: (pdb_path, scores_dict) where:
            - pdb_path (str): Path to predicted complex structure (PDB format)
            - scores_dict (dict): Confidence metrics including pLDDT, PAE, pTM, iPTM
    """

    input_json_data = create_input_dict(
        binder_seq, target_seq, binder_chain, target_chains, design_name, seed
    )
    return _run_af3(
        input_json_data,
        output_dir,
        binder_chain=binder_chain,
        msa_mode=msa_mode,
        run_settings=run_settings,
    )


def main(
    input_json: str,
    output_dir: str,
    msa_db_dir: str,
    binder_chain: str,
    msa_mode: str,
    af3_repo_path: str,
    af3_sif_path: str,
    af3_model_dir: str,
    af3_db_dir: str,
    use_metagenomic_db: bool,
):
    print("Running AF3...")

    results = pd.DataFrame(columns=["name", "plddt", "iptm"])
    with open(input_json) as f:
        input_json_data = json.load(f)
        if isinstance(input_json_data, list):
            print(
                f"Detected list input with {len(input_json_data)} items. Running AF3 for each."
            )
            for i, input_json_dict in enumerate(input_json_data):
                run_settings = {
                    "msa_db_dir": msa_db_dir,
                    "af3_repo_path": af3_repo_path,
                    "af3_sif_path": af3_sif_path,
                    "af3_model_dir": af3_model_dir,
                    "af3_db_dir": af3_db_dir,
                    "use_metagenomic_db": use_metagenomic_db,
                }
                pdb_path, scores = _run_af3(
                    input_json_dict,
                    output_dir,
                    binder_chain=binder_chain,
                    msa_mode=msa_mode,
                    run_settings=run_settings,
                )
                results = results.append(
                    {
                        "name": input_json_dict["name"],
                        "plddt": scores["plddt"],
                        "iptm": scores["iptm"],
                    },
                    ignore_index=True,
                )
                # save pae matrix
                pae_matrix_path = os.path.join(
                    output_dir, f"{input_json_dict['name']}_pae_matrix.npy"
                )
                np.save(pae_matrix_path, scores["pae_matrix"])

                print(
                    f"Folded {i + 1}/{len(input_json_data)} AF3 structures: design {input_json_dict['name']}, iptm {scores['iptm']}, plddt {scores['plddt']}"
                )
        else:
            run_settings = {
                "msa_db_dir": msa_db_dir,
                "af3_repo_path": af3_repo_path,
                "af3_sif_path": af3_sif_path,
                "af3_model_dir": af3_model_dir,
                "af3_db_dir": af3_db_dir,
                "use_metagenomic_db": use_metagenomic_db,
            }
            pdb_path, scores = _run_af3(
                input_json_data,
                output_dir,
                binder_chain=binder_chain,
                msa_mode=msa_mode,
                run_settings=run_settings,
            )
            results = results.append(
                {
                    "name": input_json_data["name"],
                    "plddt": scores["plddt"],
                    "iptm": scores["iptm"],
                },
                ignore_index=True,
            )
            # save pae matrix
            pae_matrix_path = os.path.join(
                output_dir, f"{input_json_data['name']}_pae_matrix.npy"
            )
            np.save(pae_matrix_path, scores["pae_matrix"])

    results.to_csv(os.path.join(output_dir, "results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        "-i",
        required=True,
        help="Path to input json file. Must be absolute",
    )
    parser.add_argument(
        "--output_dir", "-o", required=True, help="Output directory for AF3 outputs."
    )
    parser.add_argument(
        "--msa_db_dir",
        "-d",
        required=False,
        help="Path to MSA database directory for ColabFold.",
    )
    parser.add_argument(
        "--binder_chain",
        "-b",
        required=False,
        default="B",
        help="Chain ID of the binder chain.",
    )
    parser.add_argument(
        "--msa_mode",
        "-m",
        required=False,
        default="colabfold",
        help="MSA mode. Can be either 'local' or 'colabfold' or 'target' for target only (colabfold default).",
    )
    parser.add_argument(
        "--af3_repo_path", required=False, help="Path to local AlphaFold3 repo to bind."
    )
    parser.add_argument(
        "--af3_sif_path",
        required=False,
        help="Path to the AlphaFold3 Singularity image (.sif).",
    )
    parser.add_argument(
        "--af3_model_dir", required=False, help="Path to AF3 model weights directory."
    )
    parser.add_argument(
        "--af3_db_dir", required=False, help="Path to AF3 public databases directory."
    )
    parser.add_argument(
        "--use_metagenomic_db",
        action="store_true",
        help="Use metagenomic database for MSA generation.",
    )
    args = parser.parse_args()

    main(
        input_json=args.input_json,
        output_dir=args.output_dir,
        msa_db_dir=args.msa_db_dir,
        binder_chain=args.binder_chain,
        msa_mode=args.msa_mode,
        af3_repo_path=args.af3_repo_path,
        af3_sif_path=args.af3_sif_path,
        af3_model_dir=args.af3_model_dir,
        af3_db_dir=args.af3_db_dir,
        use_metagenomic_db=args.use_metagenomic_db,
    )
