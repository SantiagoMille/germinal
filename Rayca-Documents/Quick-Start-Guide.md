# Germinal Quick Start Guide

## Prerequisites
- Docker with GPU support (NVIDIA Container Toolkit)
- NVIDIA GPU with 40GB+ VRAM
- ~50GB disk space

## 1. Build the Image (First Time Only)
```bash
cd /path/to/germinal
docker compose -f Rayca-Code/docker-compose.yml build germinal
```
Build time: ~15-30 minutes (downloads PyRosetta 1.7GB, AF params 3GB)

## 2. Start Container
```bash
docker compose -f Rayca-Code/docker-compose.yml run --rm germinal bash
```

## 3. Verify Installation (Inside Container)
```bash
python validate_install.py
```

## 4. Run Default Pipeline
```bash
python run_germinal.py
```
This runs nanobody design against PD-L1 target using Chai structure predictor.

## 5. Customize Your Run
```bash
# Different target
python run_germinal.py target=il3

# ScFv instead of nanobody
python run_germinal.py run=scfv filter/initial=scfv filter/final=scfv

# Adjust parameters
python run_germinal.py max_trajectories=100 weights_plddt=1.5
```

## 6. View Results
Results saved to `results/` directory (mounted to host):
- `accepted/` - Passing designs
- `all_trajectories.csv` - Complete metrics

## Common Issues

**No GPU detected**: Ensure NVIDIA Container Toolkit is installed and Docker has GPU access.

**Out of memory**: Reduce batch size or use GPU with more VRAM (40GB+ recommended).

**Slow performance**: First run downloads model weights. Subsequent runs are faster.

## Next Steps
- See [Readme-EndUser.md](Readme-EndUser.md) for detailed configuration
- See [Readme-Administrator.md](Readme-Administrator.md) for deployment options
- Check original [README.md](../README.md) for scientific background
