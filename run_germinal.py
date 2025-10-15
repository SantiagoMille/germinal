"""
Run Germinal for Antibody design.
"""

import time
from omegaconf import DictConfig
import hydra
import pyrosetta as pr
import numpy as np

from germinal.design.design import germinal_design
from germinal.filters import filter_utils, redesign
from germinal.utils import utils, config
from germinal.utils.io import Trajectory


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # process hydra configuration
    processed_cfg = config.process_config(cfg)
    # get the individual settings
    prelim_run_settings, target_settings, initial_filters, final_filters = (
        processed_cfg.values()
    )
    # initialize run. adds design_path to run_settings
    io, run_settings = config.initialize_germinal_run(
        prelim_run_settings, target_settings
    )

    io.save_run_config(run_settings, target_settings)
    # initialize pyrosetta
    pr.init(
        f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
        f"-holes:dalphaball {run_settings['dalphaball_path']} "
        f"-corrections::beta_nov16 true -relax:default_repeats 1"
    )
    print(f"============================\nExperiment name: {run_settings['experiment_name']}\n============================")
    print(f"Processed config: {target_settings}\n{initial_filters}\n{final_filters}")

    # Initialize pre-set seeds for experiments if desired
    if run_settings["pregenerate_seeds"]:
        # Pre-generate seeds for reproducibility
        np.random.seed(1)
        run_seeds = [
            np.random.randint(2**32 - 1)
            for _ in range(run_settings["max_trajectories"])
        ]
    else:
        # Use a random initial seed for this run
        init_seed = int(time.time_ns()) % (2**32 - 1)
        print(f"Initial seed: {init_seed}")
        np.random.seed(init_seed)
    
    start_time = time.time()
    failed_design = 0
    num_accepted = 0
    num_failed = 0
    
    # =================================== Germinal hallucination loop
    for i in range(run_settings["max_trajectories"]):
        # Check termination conditions
        terminate, reason = io.check_termination_conditions(
            run_settings, n_trajectories=i
        )
        if terminate:
            print(reason)
            break

        # set seed for trajectory
        trajectory_start_time = time.time()
        if run_settings["pregenerate_seeds"]:
            seed = run_seeds[i]
        else:
            seed = int(np.random.randint(0, 999999, size=1, dtype=int)[0])

        # Unique design name
        design_name = f"{target_settings['target_name']}_{run_settings['type']}_s{seed}"

        if io.check_existing_seed(seed):
            print(f"Trajectory {i} with seed {seed} already exists. Trying new seed.")
            continue

        trajectory = Trajectory(
            design_name,
            run_settings["experiment_name"],
            ",".join(map(str, run_settings["cdr_lengths"])),
            target_settings["target_hotspots"],
        )
        trajectory.set_save_location("trajectories")

        print(f"\nStarting trajectory {i + 1}: {design_name}")

        # Germinal design function
        design_output = germinal_design(
            design_name, run_settings, target_settings, io, seed
        )

        # retrieve hallucination status
        trajectory_metrics_last = utils.copy_dict(design_output.aux["log"])  # final log
        hallucination_success = trajectory_metrics_last.get("terminate", "") == ""
        # if hallucination failed, continue to next trajectory
        if not hallucination_success:
            trajectory_time = utils.get_clean_time(time.time(), trajectory_start_time)
            print(f"Trajectory took: {trajectory_time}\n")
            failed_design += 1
            num_failed += 1
            continue

        # retrieve hallucination metrics
        trajectory_metrics = utils.copy_dict(design_output._tmp["best"]["log"])
        trajectory.update_trajectory_metrics(trajectory_metrics)
        trajectory_sequence = design_output._tmp["best"]["seq"]
        target_len = design_output._target_len
        trajectory_pdb_af = str(
            io.layout.trajectories / f"structures/{design_name}.pdb"
        )
        trajectory.update_other_metrics(
            {
                "trajectory_pdb_af": trajectory_pdb_af,
                "trajectory_sequence": trajectory_sequence,
                "target_len": target_len,
            }
        )
        trajectory.set_save_location("trajectories")
        # calculate time for trajectory
        trajectory_time = utils.get_clean_time(time.time(), trajectory_start_time)
        print(f"Trajectory took: {trajectory_time}\n")

        # ====================================================================================
        # First filter check - cofold and check basic structural filters
        # ====================================================================================
        print("Running initial cofolding filters")
        filter_metrics, filter_results, pass_initial_filters, final_struct = (
            filter_utils.run_filters(
                trajectory,
                run_settings,
                target_settings,
                initial_filters,
                io,
                trajectory_sequence,
                trajectory_pdb_af,
                target_len
            )
        )

        utils.clear_memory(clear_jax=False)
        if not pass_initial_filters:
            # save trajectory as not passing initial cofolding filters
            print("Trajectory not passing initial cofolding filters, skipping to next trajectory")
            complete_filter_data = {**filter_metrics, **filter_results}
            trajectory.update_filtering_metrics(complete_filter_data)
            trajectory.set_final_struct(final_struct)
            trajectory.update_other_metrics(
                {
                    "design_time": utils.get_clean_time(
                        time.time(), trajectory_start_time
                    ),
                }
            )
            trajectory.set_save_location("trajectories")
            io.save_trajectory(trajectory)
            num_failed += 1
            continue

        # ====================================================================================
        # AbMPNN redesign
        # ====================================================================================
        print("\nStarting AbMPNN redesign...\n")
        abmpnn_sequences, abmpnn_success = redesign.run_abmpnn_redesign_pipeline(
            trajectory_pdb_af=trajectory_pdb_af,
            target_chain=target_settings["target_chain"],
            binder_chain=target_settings["binder_chain"],
            run_settings=run_settings,
            atom_distance_cutoff=run_settings["atom_distance_cutoff"],
        )

        if not abmpnn_success:
            print("MPNN redesign failed, skipping to next trajectory")
            continue

        # ====================================================================================
        # Final filter check - run filters on MPNN redesigned sequences
        # ====================================================================================

        # Process MPNN redesigned sequences
        if len(abmpnn_sequences) > 0:
            for j, abmpnn_sequence in enumerate(abmpnn_sequences):
                mpnn_trajectory = trajectory.copy()
                mpnn_trajectory.rename(f"{design_name}_abmpnn_{j + 1}")
                
                # run final set of filters on AbMPNN redesigned sequences
                print("Running final filters on AbMPNN redesigned sequences")
                filter_metrics, filter_results, accepted, final_struct = (
                    filter_utils.run_filters(
                        mpnn_trajectory,
                        run_settings,
                        target_settings,
                        final_filters,
                        io,
                        abmpnn_sequence["seq"],
                        trajectory_pdb_af,
                        target_len
                    )
                )
                # save trajectory
                design_time = utils.get_clean_time(time.time(), trajectory_start_time)
                complete_filter_data = {**filter_metrics, **filter_results}
                mpnn_trajectory.update_filtering_metrics(complete_filter_data)
                mpnn_trajectory.set_final_struct(final_struct)
                mpnn_trajectory.update_other_metrics(
                    {
                        "trajectory_sequence": abmpnn_sequence["seq"],
                        "design_time": design_time,
                        "abmpnn_score": abmpnn_sequence["score"],
                        "abmpnn_seqid": abmpnn_sequence["seqid"],
                    }
                )
                mpnn_trajectory.set_save_location(
                    "accepted" if accepted else "redesign_candidates"
                )
                io.save_trajectory(mpnn_trajectory)
                if accepted:
                    print(f"========================================")
                    print(f"Design {mpnn_trajectory.design_name} accepted!")
                    print(f"========================================")
                    num_accepted += 1
                else:
                    num_failed += 1
                utils.clear_memory(clear_jax=False)
    
    # print and save final run summary
    total_runtime = utils.get_clean_time(time.time(), start_time)
    run_summary = f"Finished all designs after {i + 1} attempted trajectories.\n" \
                  f"{failed_design} designs failed initial Germinal design.\n" \
                  f"{num_failed} designs failed filters and were rejected.\n" \
                  f"{num_accepted} designs passed all filters and were accepted.\n" \
                  f"Elapsed: {total_runtime}."
    print(run_summary)
    with open(str(io.layout.root / "run_summary.txt"), "w") as f:
        f.write(run_summary)


if __name__ == "__main__":
    main()
