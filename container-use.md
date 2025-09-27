# Germinal Container Usage

This document describes running Germinal via Docker or Docker Compose, including GPU prerequisites, volume mounts, and common run patterns.

## Prerequisites

- Docker 24+
- NVIDIA GPU with a recent driver (CUDA 12+ recommended)
- NVIDIA Container Toolkit installed and configured
  - See `https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html`

Verify GPU access for containers:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 nvidia-smi
```

## Build the Image

```bash
# From repo root
docker build -t germinal:latest .

# Or with Compose (uses docker-compose.yml)
docker compose build
```

## Volumes and Data Locations

- `/workspace/params` – AlphaFold-Multimer params (baked into the image by default)
- `/workspace/pdbs` – Input PDBs
- `/workspace/runs` – Outputs (mount to your host to persist)

Create host dirs as needed:

```bash
mkdir -p pdbs runs
```

## Recommended Run Pattern

Decide your host input/output locations, then mount them explicitly.

- Host input PDB: `/abs/path/to/your_target.pdb`
- Host output dir: `/abs/path/to/run_outputs`
- Settings: use repo defaults inside the container or mount your own configs

### 1) Use repository defaults for settings (inside container), mount only outputs

```bash
mkdir -p /abs/path/to/run_outputs

# Default VHH + PD-L1 + default filters
# Outputs go to /abs/path/to/run_outputs/germinal_run

docker run --gpus all --rm -it \
  --ulimit nofile=65536:65536 \
  -v /abs/path/to/run_outputs:/workspace/runs \
  -v "$PWD/pdbs:/workspace/pdbs:ro" \
  -w /workspace germinal:latest \
  python run_germinal.py project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

scFv variant:

```bash
docker run --gpus all --rm -it \
  --ulimit nofile=65536:65536 \
  -v /abs/path/to/run_outputs:/workspace/runs \
  -v "$PWD/pdbs:/workspace/pdbs:ro" \
  -w /workspace germinal:latest \
  python run_germinal.py run=scfv project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

### 2) Mount a custom target YAML and input PDB from the host

```bash
mkdir -p /abs/path/to/run_outputs

docker run --gpus all --rm -it \
  --ulimit nofile=65536:65536 \
  -v /abs/path/to/run_outputs:/workspace/runs \
  -v /abs/path/to/my_target.yaml:/workspace/configs/target/my_target.yaml:ro \
  -v /abs/path/to/your_target.pdb:/workspace/pdbs/your_target.pdb:ro \
  -w /workspace germinal:latest \
  python run_germinal.py target=my_target project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

### 3) Mount custom filter/run settings too

```bash
docker run --gpus all --rm -it \
  --ulimit nofile=65536:65536 \
  -v /abs/path/to/run_outputs:/workspace/runs \
  -v /abs/path/to/my_target.yaml:/workspace/configs/target/my_target.yaml:ro \
  -v /abs/path/to/my_initial.yaml:/workspace/configs/filter/initial/custom.yaml:ro \
  -v /abs/path/to/my_final.yaml:/workspace/configs/filter/final/custom.yaml:ro \
  -v /abs/path/to/your_target.pdb:/workspace/pdbs/your_target.pdb:ro \
  -w /workspace germinal:latest \
  python run_germinal.py target=my_target filter.initial=custom filter.final=custom \
    project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

Notes:
- Always mount your output directory to `/workspace/runs` and set `project_dir=/workspace/runs results_dir=. experiment_name=...` to route outputs to the mount.
- Increase file descriptor limits with `--ulimit nofile=65536:65536` for heavy workloads.
- The image includes AF-Multimer params in `/workspace/params` and PyRosetta via conda.

## Quick Starts

Interactive shell:

```bash
docker run --rm -it --gpus all \
  --ulimit nofile=65536:65536 \
  -v "$PWD/pdbs:/workspace/pdbs" \
  -v "$PWD/runs:/workspace/runs" \
  -w /workspace germinal:latest bash
```

Inside the container, run:

```bash
python run_germinal.py project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

Compose run (no shell):

```bash
docker compose run --rm germinal python run_germinal.py \
  project_dir=/workspace/runs results_dir=. experiment_name=germinal_run
```

Optional shared memory toggles:
- If you encounter shared-memory or DataLoader/NCCL-related errors, add `--shm-size=16g` to `docker run` (sets host RAM allocation, not GPU VRAM), and/or `--ipc=host`.
- For Compose, you can set `shm_size: 16gb` and/or `ipc: host` in `docker-compose.yml`.


## Troubleshooting

- No outputs under your host `runs` folder:
  - Ensure you set `project_dir=/workspace/runs results_dir=. experiment_name=...` in the command.
  - Verify your `-v /host/path:/workspace/runs` mount.
- EntryPoint errors:
  - Rebuild the image to pick up the latest entrypoint.
- GPU not visible:
  - Confirm `nvidia-smi` works on host and the NVIDIA Container Toolkit is installed.
