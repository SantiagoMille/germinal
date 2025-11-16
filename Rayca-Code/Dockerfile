# Germinal Containerization - Enhanced Dockerfile
# Based on original Dockerfile with additional version pinning for reproducibility
# Date: 2025-11-16

# GPU runtime base with CUDA 12.4 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

LABEL maintainer="Rayca Containerization Project"
LABEL description="Germinal: Epitope-targeted de novo antibody design"
LABEL version="1.0.0"

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# System dependencies with version pinning where critical
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    bzip2 \
    aria2 \
    ffmpeg \
    procps \
    tini \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Install micromamba for lightweight conda environment management
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_BANNER=1
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create conda environment with Python 3.10
ENV CONDA_ENV=germinal
RUN micromamba create -y -n ${CONDA_ENV} python=3.10 && micromamba clean -a -y

ENV PATH=/opt/conda/envs/${CONDA_ENV}/bin:${PATH}
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg
ENV HYDRA_FULL_ERROR=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /workspace

# Copy essential files first for better layer caching
COPY setup.py README.md ./
COPY colabdesign /workspace/colabdesign

# Version pinning for PyTorch ecosystem
ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG CUDA_SUFFIX=cu124
ARG JAX_CUDA=true
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${CUDA_SUFFIX}

# Install core Python packages with version pinning
RUN set -euxo pipefail; \
  python -m pip install --upgrade pip==24.0 wheel==0.42.0 setuptools==69.5.1; \
  python -m pip install uv==0.1.24; \
  # Core scientific packages
  uv pip install --system \
    pandas==2.2.2 \
    matplotlib==3.8.5 \
    numpy==1.26.4 \
    biopython==1.83 \
    scipy==1.13.1 \
    seaborn==0.13.2 \
    tqdm==4.66.4 \
    ffmpeg-python==0.2.0 \
    py3dmol==2.1.0 \
    chex==0.1.86 \
    dm-tree==0.1.8 \
    joblib==1.4.2 \
    ml-collections==0.1.1 \
    immutabledict==4.2.0 \
    optax==0.2.2 \
    cvxopt==1.3.2 \
    mdtraj==1.9.9 \
    colabfold==1.5.5; \
  # Install local ColabDesign
  uv pip install --system -e /workspace/colabdesign; \
  true

# Install PyTorch ecosystem
RUN set -euxo pipefail; \
  uv pip install --system --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"; \
  uv pip install --system torchtyping==0.1.5; \
  # PyTorch Geometric
  if [ "${CUDA_SUFFIX}" = "cpu" ]; then \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"; \
  else \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}%2B${CUDA_SUFFIX}.html"; \
  fi; \
  uv pip install --system "torch_geometric==2.6.1" -f "${PYG_URL}"; \
  # IgLM and Chai-lab
  uv pip install --system iglm==1.0.1 chai-lab==0.6.1; \
  true

# Install JAX with CUDA support (pinned version)
RUN set -euxo pipefail; \
  uv pip install --system jax==0.5.3; \
  uv pip install --system dm-haiku==0.0.13; \
  uv pip install --system hydra-core==1.3.2 omegaconf==2.3.0; \
  if [ "${JAX_CUDA}" = "true" ]; then \
    uv pip install --system "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
  fi; \
  true

# Install PyRosetta via pyrosetta-installer (requires network access)
# This will download the appropriate PyRosetta wheel for Python 3.10
RUN set -euxo pipefail; \
  uv pip install --system pyrosetta-installer==0.2.1; \
  python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"; \
  true

# Copy remaining source code
COPY . /workspace
RUN uv pip install --system -e .

# Download AlphaFold-Multimer parameters (~3GB compressed, ~10GB uncompressed)
# This makes the image standalone and HPC-friendly
RUN set -euxo pipefail; \
  mkdir -p /workspace/params; \
  cd /workspace/params; \
  aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar; \
  tar -xf alphafold_params_2022-12-06.tar -C /workspace/params; \
  rm -f alphafold_params_2022-12-06.tar; \
  chmod -R a+rX /workspace/params

# Create required directories
RUN mkdir -p /workspace/pdbs /workspace/results

# Verify critical installations
RUN python -c "import germinal; import colabdesign; import jax; print(f'JAX backend: {jax.default_backend()}')"

# Default command
CMD ["bash"]
