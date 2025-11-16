# Initial Mega Prompt - Germinal Containerization

Date: 2025-11-16

## Goal
Dockerize Germinal tool with minimum impact on user flow compared to official instructions, using Docker Compose for easy end-user operation.

## Directory Structure Requirements

### Rayca-Code/
- Docker Compose file (V2 syntax: `docker compose`, NOT `docker-compose`)
- Dockerfile (evaluate existing one first, replace if not well-structured)
- Helper scripts and additional code as needed
- Version pinning for reproducibility

### Rayca-Documents/
- `Readme-EndUser.md` - For users without Docker background
- `Readme-Administrator.md` - For system administrators and DevOps engineers
- `Quick-Start-Guide.md` - Maximum 100 lines (complex tool)
- `Container-Tests/` subdirectory with numbered Jupyter notebooks
- Bug fix log if issues found
- This `InitialMegaPrompt.md` file

## Technical Specifications

### Base Image
- NVIDIA CUDA with Ubuntu 22.04 underlay
- Source: https://hub.docker.com/r/nvidia/cuda/tags

### PyRosetta
- Use from `rosettacommons/rosetta:serial` image (multi-stage build)
- Non-commercial license for academic use
- Pre-installed at `/usr/local/lib/python3.8/dist-packages/pyrosetta/` (3.7GB)
- Version: Python 3.8 compatible

### Model Weights
- Include AlphaFold-Multimer params in Docker image (~3GB compressed, ~10GB uncompressed)
- Download from: https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
- Makes image HPC cluster friendly (standalone)
- Expected final image size: 15-20GB
- Expected build time: 15-30 minutes

### GPU Requirements
- Document 40GB+ VRAM requirement (GPU only, no CPU fallback)
- Tested on: A100 40GB, H100 40GB MIG, L40S 48GB, A100 80GB, H100 80GB

## Workflow

### Git Management
1. Create `Containerization-Dev` branch from `ipsae` branch (has latest features including ipSAE scoring)
2. Commit files one by one frequently for trackability
3. Keep `CLAUDE.md` updated with latest findings

### Development Approach
1. Avoid modifying original files/directories when possible
2. Follow official installation methods unless bugs occur (then ask for approval)
3. Use Docker Compose V2 syntax (no `version:` field, start with `services:`)
4. Pin versions for reproducibility

### Testing Strategy (in Container-Tests/)
1. **Essential Container Functionality Tests**
   - Python imports work
   - CUDA/GPU accessible
   - PyRosetta initializes
   - ColabDesign loads
   - JAX uses GPU backend

2. **Full Pipeline Validation Tests**
   - Complete Germinal pipeline execution
   - All stages: Hallucination → Initial Filters → AbMPNN Redesign → Final Filters
   - Verify output structure and metrics

### Test File Organization
- Numbered Jupyter notebooks: `01_`, `02_`, etc.
- Executed versions: `*_executed.ipynb`
- Markdown files explaining expected outputs
- Separate test files per mode/scenario if needed

### Bug Documentation
- Log all fixes in dedicated markdown file
- Include rationales and ask for approval before modifying tool behavior
- Example format: `/home/azureuser/RFantibody/Rayca-Documents/RFantibody-Bug-Fixes-Log.md`

## Documentation Guidelines

### General
- Only include verified, tested information
- Draw inspiration from RFantibody examples but don't copy unverified content
- Documentation is LAST task (after all testing complete)
- Use date format: YYYY-MM-DD (ISO format)
- Naming convention: Word1-RelevantToWord1IfNeeded_Word2-RelevantToWord2IfNeeded

### Readme-EndUser.md Audience
- Users without Docker background
- First-time tool users
- Step-by-step instructions

### Readme-Administrator.md Audience
- System administrators
- DevOps engineers
- Technical deep-dive

### Quick-Start-Guide.md
- Maximum 100 lines (complex tool)
- Minimal steps to get running
- Reference detailed docs for more info

## API Keys and Credentials
- If required for model downloads, prepare specific file/variable locations
- Document exact location for user to provide credentials
- AF3 weights require DeepMind request (optional for this containerization)

## Important Reminders
- All tests run INSIDE container, not on host VM
- Ask approval before installing anything locally on host
- When todo list is complete, list possible next steps for user
- Keep notes for second test phase (tool-specific) to avoid forgetting when context compresses

## Reference Documents
- Example documentation structure: `/home/azureuser/RFantibody/Rayca-Documents/`
- Bug fix log example: `/home/azureuser/RFantibody/Rayca-Documents/RFantibody-Bug-Fixes-Log.md`
