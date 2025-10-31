"""Germinal I/O Management Module

This module provides comprehensive input/output management for the Germinal protein
design system. It handles directory structure creation, trajectory data management,
file operations with concurrency safety, and result organization.

The module implements:
- Directory structure creation and management (RunLayout)
- Trajectory data storage and organization (Trajectory)
- Thread-safe file operations with locking mechanisms
- CSV-based data persistence with automatic column handling
- Failure tracking and termination condition monitoring

Key Classes:
    RunLayout: Defines and creates directory structure for design runs
    Trajectory: Stores and manages individual trajectory information
    IO: Main I/O handler with thread-safe operations

Key Features:
    - Automatic directory structure creation
    - Thread-safe CSV operations using file locks
    - Trajectory filtering and organization
    - Failure count tracking and analysis
    - Termination condition monitoring

Dependencies:
    - pandas for data manipulation
    - filelock for thread-safe file operations
    - pathlib for cross-platform path handling
"""

from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import yaml
import pandas as pd
from typing import Dict, Any
from filelock import FileLock


TRAJECTORY_METRICS_TO_SAVE = [
    "plddt",
    "ptm",
    "i_ptm",
    "i_pae",
    "pae",
    "loss",
    "ablm_ll",
    "helix",
    "beta_strand",
]


@dataclass
class RunLayout:
    """Directory structure definition and creation for Germinal design runs.
    
    This dataclass defines the standardized directory structure used by Germinal
    for organizing design run outputs, including trajectories, candidates, accepted
    designs, logs, and summary files. Provides methods for creating the complete
    directory hierarchy with proper initialization.
    
    Attributes:
        root (Path): Root directory for the design run
        trajectories (Path): Directory for trajectory data and structures
        redesign_candidates (Path): Directory for redesign candidate structures
        accepted (Path): Directory for accepted final designs
        logs (Path): Directory for run logs and debugging information
        all_csv (Path): Path to comprehensive trajectory summary CSV
        failure_csv (Path): Path to failure count tracking CSV
        final_config (Path): Path to final configuration YAML file
        
    Directory Structure:
        root/
        ├── trajectories/
        │   ├── structures/
        │   ├── plots/
        │   └── designs.csv
        ├── redesign_candidates/
        │   ├── structures/
        │   ├── plots/
        │   └── designs.csv
        ├── accepted/
        │   ├── structures/
        │   ├── plots/
        │   └── designs.csv
        ├── logs/
        ├── all_trajectories.csv
        ├── failure_counts.csv
        └── final_config.yaml
    """

    root: Path
    trajectories: Path
    redesign_candidates: Path
    accepted: Path
    logs: Path
    all_csv: Path
    failure_csv: Path
    final_config: Path

    @classmethod
    def create(cls, root: Path | str):
        """Create complete directory structure for a new Germinal design run.
        
        Initializes the full directory hierarchy required for a Germinal design run,
        including all subdirectories, placeholder files, and initial CSV files.
        Creates the structure with proper permissions and initializes empty tracking files.
        
        Args:
            root (Path | str): Root directory path for the design run. Will be created
                if it doesn't exist, along with all parent directories.
                
        Returns:
            RunLayout: Initialized RunLayout instance with all paths configured
            and directory structure created.
        """
        if isinstance(root, str):
            root = Path(root)

        root.mkdir(parents=True, exist_ok=True)

        traj = root / "trajectories"
        redesign = root / "redesign_candidates"
        accepted = root / "accepted"
        logs = root / "logs"
        # Initialize empty tracking files for run management
        all_csv = root / "all_trajectories.csv"
        failure_csv = root / "failure_counts.csv"
        final_config = root / "final_config.yaml"
        all_csv.touch()
        failure_csv.touch()
        final_config.touch()

        # Create main subdirectories with internal structure
        for main_dir in (traj, redesign, accepted):
            main_dir.mkdir(parents=True, exist_ok=True)
            (main_dir / "structures").mkdir(exist_ok=True)
            (main_dir / "plots").mkdir(exist_ok=True)
            # Initialize trajectory tracking CSV for each directory
            (main_dir / "designs.csv").touch()

        logs.mkdir(parents=True, exist_ok=True)

        return cls(
            root=root,
            trajectories=traj,
            redesign_candidates=redesign,
            accepted=accepted,
            logs=logs,
            all_csv=all_csv,
            failure_csv=failure_csv,
            final_config=final_config,
        )


class Trajectory:
    """Individual trajectory data container with metrics and metadata management.
    
    This class encapsulates all information associated with a single design trajectory,
    including trajectory metrics, filtering results, structural data, and metadata.
    Provides methods for updating different metric categories and managing trajectory
    lifecycle from creation to final storage.
    
    The class organizes metrics into three categories:
    - Trajectory metrics: Core optimization metrics (pLDDT, pTM, loss, etc.)
    - Filtering metrics: Results from filtering and quality assessment
    - Other metrics: Additional analysis results and metadata
    
    Attributes:
        design_name (str): Unique identifier for this trajectory
        experiment_name (str): Name of the parent experiment
        cdr_lengths (str): CDR length specification string
        target_hotspots (str): Target hotspot residue specification
        trajectory_metrics (dict): Core trajectory optimization metrics
        filtering_metrics (dict): Quality filtering and assessment results
        other_metrics (dict): Additional analysis and metadata
        save_location (str): Target directory for saving ('trajectories', 'redesign_candidates', 'accepted')
        final_struct (str): Path to final structure file
    """

    def __init__(
        self,
        design_name: str,
        experiment_name: str,
        cdr_lengths: str,
        target_hotspots: str,
    ):
        self.design_name = design_name
        self.experiment_name = experiment_name
        self.cdr_lengths = cdr_lengths
        self.target_hotspots = target_hotspots

        self.trajectory_metrics = {}
        self.filtering_metrics = {}
        self.other_metrics = {}

        self.save_location = ""  # trajectories, redesign_candidates, or accepted
        self.final_struct = ""

    def get_trajectory(self):
        """Consolidate all trajectory information into a single dictionary.
        
        Combines trajectory metrics, filtering metrics, and other metrics with
        basic metadata into a comprehensive dictionary representation suitable
        for CSV export and analysis.
        
        Returns:
            dict: Complete trajectory information with all metrics and metadata
                combined into a flat dictionary structure.
        """
        final_trajectory_info = {
            "design_name": self.design_name,
            "experiment_name": self.experiment_name,
            "cdr_lengths": self.cdr_lengths,
            "target_hotspots": self.target_hotspots,
            **self.trajectory_metrics,
            **self.filtering_metrics,
            **self.other_metrics,
            "final_structure_path": self.final_struct
        }

        return final_trajectory_info

    def update_trajectory_metrics(self, trajectory_metrics: Dict[str, Any]):
        """Update core trajectory optimization metrics.
        
        Updates the trajectory metrics with values from the design optimization
        process. Only metrics defined in TRAJECTORY_METRICS_TO_SAVE are stored
        to maintain consistency across trajectories.
        
        Args:
            trajectory_metrics (Dict[str, Any]): Dictionary of trajectory metrics
                from the optimization process. Should include metrics like pLDDT,
                pTM, loss, etc.
        """
        for metric in TRAJECTORY_METRICS_TO_SAVE:
            if metric in trajectory_metrics:
                self.trajectory_metrics[metric] = trajectory_metrics[metric]

    def update_filtering_metrics(self, filtering_metrics: Dict[str, Any]):
        """Update quality filtering and assessment metrics.
        
        Updates metrics related to trajectory quality assessment, filtering
        criteria evaluation, and selection decisions.
        
        Args:
            filtering_metrics (Dict[str, Any]): Dictionary of filtering results
                including quality scores, pass/fail flags, and selection criteria.
        """
        self.filtering_metrics.update(filtering_metrics)

    def update_other_metrics(self, other_metrics: Dict[str, Any]):
        """Update additional analysis metrics and metadata.
        
        Updates supplementary metrics that don't fall into trajectory or filtering
        categories, such as secondary analysis results, timing information, or
        custom evaluation metrics.
        
        Args:
            other_metrics (Dict[str, Any]): Dictionary of additional metrics
                and metadata for this trajectory.
        """
        self.other_metrics.update(other_metrics)

    def set_save_location(self, save_location: str):
        """Set the target directory for trajectory storage.
        
        Specifies which directory category this trajectory should be saved to
        based on its quality and selection status.
        
        Args:
            save_location (str): Target directory name. Must be one of:
                'trajectories', 'redesign_candidates', or 'accepted'.
        """
        self.save_location = save_location

    def set_final_struct(self, final_struct: str):
        """Set the path to the final structure file for this trajectory.
        
        Args:
            final_struct (str): Path to the final PDB structure file
                representing the best result from this trajectory.
        """
        self.final_struct = final_struct

    def rename(self, new_name: str):
        """Change the design name identifier for this trajectory.
        
        Args:
            new_name (str): New unique identifier for this trajectory.
        """
        self.design_name = new_name

    def copy(self):
        """Create a deep copy of this trajectory instance.
        
        Creates a new Trajectory instance with identical data but independent
        from the original. Useful for creating variants or backups.
        
        Returns:
            Trajectory: New trajectory instance with copied data.
        """
        new_trajectory = Trajectory(
            design_name=self.design_name,
            experiment_name=self.experiment_name,
            cdr_lengths=self.cdr_lengths,
            target_hotspots=self.target_hotspots,
        )
        new_trajectory.trajectory_metrics = self.trajectory_metrics
        new_trajectory.filtering_metrics = self.filtering_metrics
        new_trajectory.other_metrics = self.other_metrics
        new_trajectory.save_location = self.save_location
        new_trajectory.final_struct = self.final_struct
        return new_trajectory


class IO:
    """Main I/O handler for Germinal design runs with thread-safe operations.
    
    This class provides comprehensive I/O management for Germinal design runs,
    including configuration persistence, trajectory saving, failure tracking,
    and termination condition monitoring. All file operations are thread-safe
    using file locks to prevent race conditions in multi-device environments.
    
    Key Features:
    - Thread-safe CSV operations with automatic column management
    - Configuration persistence and validation
    - Trajectory organization and storage
    - Failure count tracking and analysis
    - Termination condition monitoring
    - Seed tracking to prevent duplicate work
    
    Attributes:
        layout (RunLayout): Directory structure manager for the run
    """

    def __init__(self, layout: RunLayout):
        self.layout = layout

    def save_run_config(
        self, run_settings: Dict[str, Any], target_settings: Dict[str, Any]
    ):
        """Save run and target configuration to persistent YAML file.
        
        Persists the complete configuration for the design run to enable
        reproducibility and debugging. Combines both run and target settings
        into a single YAML file with proper structure.
        
        Args:
            run_settings (Dict[str, Any]): Complete run configuration parameters
            target_settings (Dict[str, Any]): Target-specific configuration parameters
        """
        with open(self.layout.final_config, "w") as f:
            yaml.dump(
                {"run_settings": run_settings, "target_settings": target_settings}, f
            )

        print(f"Run and target settings saved to {self.layout.final_config}")

    def check_existing_seed(self, seed: int):
        """Check if a design seed has already been processed in this run.
        
        Searches through all structure directories (trajectories, redesign_candidates,
        accepted) to determine if the given seed has already been used, preventing
        duplicate work in distributed environments.
        
        Args:
            seed (int): Design seed number to check for existing usage
            
        Returns:
            bool: True if seed has been used, False if seed is available for use
        """
        possible_dirs = [
            f"{self.layout.trajectories}/structures",
            f"{self.layout.redesign_candidates}/structures",
            f"{self.layout.accepted}/structures",
        ]
        for dir in possible_dirs:
            if not os.path.exists(dir):
                continue
            for fname in os.listdir(dir):
                if f"{seed}" in fname:
                    return True
        return False

    def check_termination_conditions(
        self, run_settings: Dict[str, Any], n_trajectories: int
    ):
        """Evaluate termination conditions for the design optimization loop.
        
        Checks multiple termination criteria including maximum accepted designs,
        maximum hallucinated trajectories, and total trajectory limits to determine
        if the optimization loop should continue or terminate.
        
        Args:
            run_settings (Dict[str, Any]): Run configuration containing termination limits:
                - max_passing_designs: Maximum number of accepted designs
                - max_hallucinated_trajectories: Maximum successful trajectories
                - max_trajectories: Maximum total trajectory attempts
            n_trajectories (int): Current number of trajectory attempts
            
        Returns:
            tuple: (should_terminate, termination_reason) where:
                - should_terminate (bool): Whether to terminate the optimization loop
                - termination_reason (str): Human-readable reason for termination
        """
        # check accepted designs
        accepted_csv_path = self.layout.accepted / "designs.csv"
        if (
            not os.path.exists(accepted_csv_path)
            or accepted_csv_path.stat().st_size == 0
        ):
            n_accepted_trajectories = 0
        else:
            n_accepted_trajectories = len(pd.read_csv(accepted_csv_path))
        if n_accepted_trajectories >= run_settings["max_passing_designs"]:
            return True, "Max accepted trajectories reached. Exiting Loop."
        # check hallucinated designs
        trajectories_csv_path = self.layout.all_csv
        if (
            not os.path.exists(trajectories_csv_path)
            or trajectories_csv_path.stat().st_size == 0
        ):
            n_hallucinated_trajectories = 0
        else:
            n_hallucinated_trajectories = len(pd.read_csv(trajectories_csv_path))
        if n_hallucinated_trajectories >= run_settings["max_hallucinated_trajectories"]:
            return True, "Max hallucinated trajectories reached. Exiting Loop."
        # check total trajectories
        if n_trajectories >= run_settings["max_trajectories"]:
            return True, "Max trajectories reached. Exiting Loop."
        return False, ""

    def update_failures(self, fail_column: str) -> None:
        """Thread-safe increment of failure counts in tracking CSV.
        
        Increments failure counts for specific failure types using thread-safe
        file operations. Creates the CSV file and columns as needed. Supports
        both single column updates and batch dictionary updates.
        
        Args:
            fail_column (str or dict): Either a column name to increment by 1,
                or a dictionary mapping column names to increment values.
        """
        if self.layout.failure_csv is None:
            return

        lock = FileLock(f"{str(self.layout.failure_csv)}.lock")
        with lock:
            # Try to read the CSV, or create an empty DataFrame if it doesn't exist or is empty
            try:
                failure_df = pd.read_csv(self.layout.failure_csv)
            except Exception:
                failure_df = pd.DataFrame()

            def strip_model_prefix(name: str) -> str:
                parts = name.split("_")
                return "_".join(parts[1:]) if parts and parts[0].isdigit() else name

            # Prepare update dict
            if isinstance(fail_column, dict):
                updates = {}
                for k, v in fail_column.items():
                    col = strip_model_prefix(k)
                    updates[col] = updates.get(col, 0) + v
            else:
                col = strip_model_prefix(fail_column)
                updates = {col: 1}

            # If DataFrame is empty, create a single row with the updates
            if failure_df.empty:
                failure_df = pd.DataFrame(
                    [{col: count for col, count in updates.items()}]
                )
            else:
                # If DataFrame has no rows, add a row
                if len(failure_df) == 0:
                    failure_df = pd.DataFrame(
                        [{col: count for col, count in updates.items()}]
                    )
                else:
                    # Increment columns, create if missing
                    for col, count in updates.items():
                        if col in failure_df.columns:
                            failure_df.loc[:, col] = (
                                failure_df[col].fillna(0).astype(int) + count
                            )
                        else:
                            failure_df[col] = count

            failure_df.to_csv(self.layout.failure_csv, index=False)

    def save_trajectory(self, trajectory: Trajectory) -> int:
        """Save trajectory data and structure with thread-safe operations.
        
        Saves trajectory information to the appropriate directory based on its
        save_location attribute. Handles both structure file copying and CSV
        data persistence with thread-safe file locking to prevent race conditions
        in multi-device environments.
        
        The method performs the following operations:
        1. Copies structure file to appropriate directory
        2. Updates category-specific CSV file with trajectory data
        3. Updates master all_trajectories.csv with common columns
        
        Args:
            trajectory (Trajectory): Complete trajectory object containing all
                metrics, metadata, and file paths for saving.
                
        Returns:
            int: 0 on successful save operation
            
        Raises:
            ValueError: If trajectory.save_location is not one of the valid options
                ('trajectories', 'redesign_candidates', 'accepted')
        """
        trajectory_data = trajectory.get_trajectory()
        trajectory_df = pd.DataFrame([trajectory_data])

        # Determine structure and CSV paths
        if trajectory.save_location == "trajectories":
            structure_path = (
                self.layout.trajectories
                / "structures"
                / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.trajectories / "designs.csv"
        elif trajectory.save_location == "redesign_candidates":
            structure_path = (
                self.layout.redesign_candidates
                / "structures"
                / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.redesign_candidates / "designs.csv"
        elif trajectory.save_location == "accepted":
            structure_path = (
                self.layout.accepted / "structures" / f"{trajectory.design_name}.pdb"
            )
            csv_path = self.layout.accepted / "designs.csv"
        else:
            raise ValueError(f"Invalid save location: {trajectory.save_location}")

        print(f"Saving trajectory structure to {structure_path}")
        print(f"Saving trajectory data to {csv_path}")

        # Save structure to appropriate directory (structure files are not locked, assumed unique per design)
        shutil.copy(trajectory.final_struct, structure_path)

        # Save trajectory data to the appropriate CSV with a lock
        csv_lock_path = f"{str(csv_path)}.lock"
        with FileLock(csv_lock_path):
            write_header = not os.path.exists(csv_path) or csv_path.stat().st_size == 0
            trajectory_df.to_csv(csv_path, mode="a", header=write_header, index=False)

        # Save/update the common all_trajectories.csv with a lock
        all_csv_lock_path = f"{str(self.layout.all_csv)}.lock"
        with FileLock(all_csv_lock_path):
            first_entry = (
                not os.path.exists(self.layout.all_csv)
                or self.layout.all_csv.stat().st_size == 0
            )
            if first_entry:
                # If first entry, just write the current row
                trajectory_df.to_csv(
                    self.layout.all_csv, mode="w", header=True, index=False
                )
            else:
                # Only keep columns common to both the new row and the existing file
                existing_df = pd.read_csv(self.layout.all_csv)
                common_columns = [c for c in trajectory_df.columns if c in existing_df.columns]
                if not common_columns:
                    # If no common columns, just append all columns of the new row
                    updated_all_trajectories_df = pd.concat(
                        [existing_df, trajectory_df], ignore_index=True, sort=False
                    )
                else:
                    # Only keep common columns for both
                    trajectory_df_common = trajectory_df[list(common_columns)]
                    existing_df_common = existing_df[list(common_columns)]
                    updated_all_trajectories_df = pd.concat(
                        [existing_df_common, trajectory_df_common], ignore_index=True
                    )
                updated_all_trajectories_df.to_csv(self.layout.all_csv, index=False)

        return 0
