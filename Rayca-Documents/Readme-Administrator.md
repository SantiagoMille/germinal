# Germinal - Administrator Guide

**Infrastructure deployment and management guide for Germinal containers**

This guide provides technical details for system administrators deploying Germinal in production environments.

---

## System Requirements

### Hardware Requirements

#### Minimum Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **RAM**: 32+ GB system memory
- **Storage**: 100+ GB available disk space
  - ~29GB for Docker image
  - Additional space for results and intermediate files
- **GPU**: NVIDIA GPU with 40GB+ VRAM (required)

#### Recommended Production Setup
- **CPU**: 16+ cores for concurrent design campaigns
- **RAM**: 64+ GB for large-scale batch processing
- **GPU**: NVIDIA GPU with 80GB VRAM (A100, H100)
  - Tested GPUs: A100 40GB, H100 40GB MIG, L40S 48GB, A100 80GB, H100 80GB
- **Storage**: NVMe SSD with 500+ GB for high-throughput campaigns
- **Network**: High-bandwidth connection (model weights downloaded during build)

### Software Requirements
- **Docker**: Version 20.10+ (Docker Compose V2)
- **NVIDIA Container Toolkit**: For GPU acceleration
- **Operating System**: Linux (Ubuntu 22.04 tested)
- **NVIDIA Driver**: Compatible with CUDA 12.4+ (535.x or newer)

---

## Container Architecture

### Image Structure
```
germinal:latest
├── Base: nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
├── Package Manager: micromamba (lightweight conda alternative)
├── Python: 3.10.19 in /opt/conda/envs/germinal
├── Core Libraries:
│   ├── PyTorch: 2.6.0+cu124
│   ├── JAX: 0.5.3 with CUDA 12 plugin
│   ├── PyRosetta: 2025.45 (installed via pyrosetta-installer)
│   ├── Chai-lab: 0.6.1
│   └── IgLM: 0.1.0
├── Germinal Code: /workspace/
├── AlphaFold Params: /workspace/params/ (~10GB)
└── Results: /workspace/results/ (bind-mounted)
```

**Size:** ~29GB
**Build Time:** 15-30 minutes
**Use Case:** Production deployments, HPC clusters

### Storage Layout
```
/path/to/germinal/
├── Dockerfile                    # Original working Dockerfile
├── Rayca-Code/
│   ├── Dockerfile                # Enhanced version (for future use)
│   └── docker-compose.yml        # V2 compose configuration
├── Rayca-Documents/
│   ├── InitialMegaPrompt.md      # Containerization requirements
│   ├── Quick-Start-Guide.md      # Quick start (60 lines)
│   ├── Readme-EndUser.md         # End user documentation
│   ├── Readme-Administrator.md   # This file
│   ├── Germinal-Bug-Fixes-Log.md # Bug fixes during containerization
│   ├── installed_packages_frozen.txt  # Exact package versions
│   └── Container-Tests/          # Test notebooks
├── configs/
│   ├── config.yaml               # Main Hydra config
│   ├── run/                      # Run parameters
│   ├── target/                   # Target protein definitions
│   └── filter/                   # Filter thresholds
├── germinal/                     # Core Python package
├── colabdesign/                  # Bundled ColabDesign library
├── pdbs/                         # Target PDB files (bind-mounted)
└── results/                      # Output directory (bind-mounted)
```

---

## Deployment Options

### Docker Compose (Recommended)
```bash
# Build
docker compose -f Rayca-Code/docker-compose.yml build germinal

# Run interactive
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash

# Run specific command
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal python run_germinal.py

# Run with Jupyter (for testing)
docker compose -f Rayca-Code/docker-compose.yml --profile jupyter up germinal-jupyter
```

### Direct Docker
```bash
# Build
docker build -t germinal .

# Run
docker run -it --rm --gpus all \
  -v "$PWD/results:/workspace/results" \
  -v "$PWD/pdbs:/workspace/pdbs" \
  germinal bash
```

### Singularity/Apptainer (HPC)
```bash
# Convert to SIF (note: 29GB image may take time)
singularity pull germinal.sif docker://germinal:latest

# Run
singularity shell --nv \
  --bind "$PWD/results:/workspace/results" \
  --bind "$PWD/pdbs:/workspace/pdbs" \
  --pwd /workspace \
  germinal.sif
```

---

## GPU Management

### Verify GPU Access
```bash
docker run --rm --gpus all germinal:latest nvidia-smi
```

### Multi-GPU Allocation
```bash
# Specific GPUs
docker run --gpus '"device=0,1"' ...

# Limit GPU memory
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -I {} kill {}
```

### Resource Monitoring
```bash
# Inside container
watch nvidia-smi

# Host monitoring
docker stats germinal
```

---

## Security Considerations

### Licenses
- **PyRosetta**: Non-commercial academic license (automatically downloaded via pyrosetta-installer)
- **AlphaFold Parameters**: DeepMind license terms apply
- **Germinal**: Apache 2.0

### Network Requirements
During build, the container downloads:
- PyPI packages (~2GB)
- PyRosetta wheel from west.rosettacommons.org (~1.7GB)
- AlphaFold-Multimer parameters from storage.googleapis.com (~3GB)

Ensure firewall allows outbound connections to:
- pypi.org
- west.rosettacommons.org
- storage.googleapis.com
- download.pytorch.org

### Container Isolation
```yaml
# docker-compose.yml security options
security_opt:
  - no-new-privileges:true
read_only: false  # Required for runtime files
```

---

## Performance Tuning

### Shared Memory
```yaml
# docker-compose.yml
shm_size: "32gb"  # Increase for large models
```

### JAX Configuration
```bash
# Pre-allocate GPU memory
XLA_PYTHON_CLIENT_PREALLOCATE=true
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

### Batch Processing
```bash
# Run multiple trajectories
python run_germinal.py max_trajectories=1000 max_passing_designs=100

# Parallel runs (different GPUs)
CUDA_VISIBLE_DEVICES=0 python run_germinal.py experiment_name=run1 &
CUDA_VISIBLE_DEVICES=1 python run_germinal.py experiment_name=run2 &
```

---

## Monitoring and Logging

### Container Logs
```bash
docker logs germinal
docker compose -f Rayca-Code/docker-compose.yml logs -f
```

### Run Monitoring
Results include:
- `run_summary.txt` - Final statistics
- `all_trajectories.csv` - Complete metrics
- `failure_counts.csv` - Failure breakdown

### Health Checks
```bash
# Inside container
python validate_install.py

# Quick import test
python -c "import germinal, pyrosetta, jax; print('OK')"
```

---

## Backup and Recovery

### Image Export
```bash
docker save germinal:latest | gzip > germinal_backup.tar.gz
```

### Image Import
```bash
gunzip -c germinal_backup.tar.gz | docker load
```

### Results Backup
```bash
# Results are in mounted volume, backup host directory
tar -czf results_backup.tar.gz results/
```

---

## Known Issues and Fixes

See [Germinal-Bug-Fixes-Log.md](Germinal-Bug-Fixes-Log.md) for documented issues.

### Common Issues
1. **matplotlib version conflict** - Use 3.8.4 (3.8.5 doesn't exist)
2. **pandas/colabfold conflict** - Use original Dockerfile (resolves automatically)
3. **PyRosetta download** - Requires network access to west.rosettacommons.org

---

## Version Information

### Frozen Package Versions
Complete list: `Rayca-Documents/installed_packages_frozen.txt`

Key versions:
- Python: 3.10.19
- PyTorch: 2.6.0+cu124
- JAX: 0.5.3
- PyRosetta: 2025.45+release
- dm-haiku: 0.0.13
- hydra-core: 1.3.2

### Git Branch
Development branch: `Containerization-Dev` (based on `ipsae`)
