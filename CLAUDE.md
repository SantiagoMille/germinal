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

# Docker
docker build -t germinal .
docker run -it --rm --gpus all -v "$PWD/results:/workspace/results" germinal bash
```

No test suite exists. Testing is done via `validate_install.py` and running the pipeline.

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
