# Germinal - End User Guide

**Get started with Germinal antibody design using Docker**

This guide provides essential commands to run the Germinal pipeline for de novo epitope-targeted antibody design.

---

## Quick Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/SantiagoMille/germinal.git
cd germinal
```

### Step 2: Build Docker Image
```bash
docker compose -f Rayca-Code/docker-compose.yml build germinal
```
**Build time:** ~15-30 minutes (downloads PyRosetta 1.7GB, AlphaFold params 3GB)

### Step 3: Start Container
```bash
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash
```

**That's it!** The container is now running with:
- All dependencies installed
- PyRosetta ready
- AlphaFold-Multimer parameters included
- GPU access configured (if available)
- Results directory mounted to host

---

## Running Germinal

### Verify Installation
```bash
python validate_install.py
```

### Default Run (Nanobody targeting PD-L1)
```bash
python run_germinal.py
```
This runs antibody hallucination with default configs (~5-10 minutes per trajectory).

### Common Configurations
```bash
# ScFv design (instead of nanobody)
python run_germinal.py run=scfv filter/initial=scfv filter/final=scfv

# Different target protein
python run_germinal.py target=il3

# Custom parameters
python run_germinal.py max_trajectories=100 weights_plddt=1.5 experiment_name=my_experiment

# Use AF3 structure predictor (requires AF3 setup)
python run_germinal.py structure_model=af3
```

### Target Configuration
Create your own target in `configs/target/`:
```yaml
target_name: "my_protein"
target_pdb_path: "pdbs/my_protein.pdb"
target_chain: "A"
binder_chain: "B"
target_hotspots: "25,26,39,41"
dimer: false
```

---

## Understanding Results

Results are saved to `results/` directory:
```
results/{target}_{type}_{timestamp}/
├── accepted/           # Designs passing all filters (SUCCESS!)
│   ├── structures/     # PDB files of accepted designs
│   └── designs.csv     # Metrics for accepted designs
├── redesign_candidates/ # Passed initial, failed final filters
├── trajectories/        # Failed initial filters
├── all_trajectories.csv # Complete metrics for all designs
└── run_summary.txt      # Final statistics
```

### Key Metrics to Review
- **external_iptm** - Interface pTM score (higher is better)
- **external_plddt** - Structure confidence (higher is better)
- **external_ipae** - Interface PAE (lower is better)
- **sc_rmsd** - Structural deviation (lower is better)
- **clashes_unrelaxed** - Number of clashes (0 is ideal)

---

## Working with Docker Compose

### Re-entering Container
```bash
# Start new container
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash
```

### Check Image
```bash
docker images germinal:latest
```

### Remove Old Image (to rebuild)
```bash
docker rmi germinal:latest
```

### Custom Volume Mounts
Edit `Rayca-Code/docker-compose.yml`:
```yaml
volumes:
  - ../results:/workspace/results
  - ../pdbs:/workspace/pdbs
  - /path/to/custom/configs:/workspace/configs
```

---

## Troubleshooting

### No GPU Detected
- Ensure NVIDIA Container Toolkit is installed
- Run with `--gpus all` flag:
  ```bash
  docker run --gpus all -it germinal:latest bash
  ```

### Out of Memory
- Germinal requires 40GB+ VRAM
- Tested GPUs: A100, H100, L40S
- Try reducing `num_models` in config

### Slow Performance
- First run downloads IgLM weights (~500MB)
- Subsequent runs are faster
- Use SSD storage for better I/O

### Build Fails
- Check internet connectivity (needs PyPI, Google Storage)
- Ensure sufficient disk space (50GB+ recommended)
- See `Rayca-Documents/Germinal-Bug-Fixes-Log.md` for known issues

---

## Next Steps

- Review [Quick-Start-Guide.md](Quick-Start-Guide.md) for essentials
- See [Readme-Administrator.md](Readme-Administrator.md) for deployment
- Check original [README.md](../README.md) for scientific details
- Explore [configs/](../configs/) for parameter tuning
