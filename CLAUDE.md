# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Install (after conda environment setup)
pip install -e .

# Run main pipeline (uses Hydra configs)
python run_germinal.py

# Run with different configs
python run_germinal.py run=scfv target=pdl1 filter/initial=scfv filter/final=scfv

# Override parameters via CLI
python run_germinal.py max_trajectories=100 weights_plddt=1.5 experiment_name=my_exp

# Validate installation
python validate_install.py

# Docker (using Docker Compose V2)
docker compose -f Rayca-Code/docker-compose.yml build germinal
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash

# Alternative: direct Docker commands
docker build -t germinal .
docker run -it --rm --gpus all -v "$PWD/results:/workspace/results" germinal bash
```

No test suite exists. Testing is done via `validate_install.py` and running the pipeline.

## Containerization Details

### Docker Image
- **Base**: NVIDIA CUDA 12.4.1 + cuDNN on Ubuntu 22.04
- **Size**: ~29GB (includes CUDA, PyRosetta 1.7GB, AF-Multimer params ~10GB)
- **Python**: 3.10.19 via micromamba
- **GPU**: Requires NVIDIA GPU with 40GB+ VRAM

### Key Package Versions (from frozen build)
- PyTorch: 2.6.0+cu124
- JAX: 0.5.3 with CUDA 12 plugin
- PyRosetta: 2025.45 (installed via pyrosetta-installer)
- Chai-lab: 0.6.1
- IgLM: 0.1.0
- pandas: 2.3.3
- Full list: `Rayca-Documents/installed_packages_frozen.txt`

### Rayca Containerization Structure
```
Rayca-Code/
├── Dockerfile                    # Enhanced Dockerfile (for future version pinning)
└── docker-compose.yml           # V2 compose file with GPU support

Rayca-Documents/
├── InitialMegaPrompt.md          # Containerization requirements
├── Germinal-Bug-Fixes-Log.md     # Bug fixes during containerization
├── installed_packages_frozen.txt  # Exact package versions from build
├── Container-Tests/              # Test notebooks
│   ├── 01_Essential-Container-Functionality.ipynb
│   └── 02_Full-Pipeline-Validation.ipynb
├── Quick-Start-Guide.md          # Quick start (max 100 lines)
├── Readme-EndUser.md             # For Docker beginners
└── Readme-Administrator.md       # For DevOps engineers
```

### Docker Compose Usage
```bash
# Build image (first time, ~15-30 minutes)
docker compose -f Rayca-Code/docker-compose.yml build germinal

# Run interactive shell
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash

# Run with Jupyter (for testing)
docker compose -f Rayca-Code/docker-compose.yml --profile jupyter up germinal-jupyter
```

## Architecture

### Pipeline Flow
1. **Hallucination** (`germinal/design/design.py`) - Generate antibody structures using ColabDesign/AF2
2. **Initial Filtering** - Apply post-hallucination filters (clashes, RMSD, hotspot proximity)
3. **AbMPNN Redesign** (`germinal/filters/redesign.py`) - Selective sequence redesign
4. **Final Filtering** - Structure prediction with AF3/Chai + comprehensive metrics

### Core Modules
- **germinal/design/** - AF2-based hallucination with custom loss functions (helix, beta, iPTM, IgLM)
- **germinal/filters/** - Structure prediction wrappers (af3.py, chai.py), metrics (pDockQ.py), PyRosetta utilities
- **germinal/utils/** - Configuration processing, IO/trajectory management, utilities
- **colabdesign/** - Bundled ColabDesign library for AF2 hallucination

### Configuration (Hydra)
```
configs/
├── config.yaml           # Main config, references defaults
├── run/                   # Run parameters (vhh.yaml, scfv.yaml)
├── target/                # Target protein definitions (pdl1.yaml, il3.yaml)
└── filter/
    ├── initial/           # Post-hallucination filters
    └── final/             # Final acceptance filters
```

Settings in `configs/run/` are in global namespace (no `run.` prefix needed for CLI overrides).

### Key Dependencies
- **PyRosetta** - Structure analysis (requires academic license)
- **JAX** - GPU-accelerated hallucination
- **PyTorch** - IgLM model
- **AF3/Chai-1** - External structure prediction (AF3 via Singularity, Chai via Python)

### Output Structure
Results are saved to `results/{target}_{type}_{timestamp}/` with:
- `accepted/` - Designs passing all filters
- `redesign_candidates/` - Passed initial, failed final filters
- `trajectories/` - Failed initial filters
- `all_trajectories.csv` - Complete metrics for all designs
