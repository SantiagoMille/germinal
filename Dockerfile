FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Germinal GPU-enabled container
# - Includes micromamba-managed conda env from environment.yml
# - Installs CUDA-enabled JAX, PyTorch, Torch Geometric, Chai, IgLM
# - Installs project and ColabDesign in editable mode
#
# Notes:
# - PyRosetta is installed automatically via conda
# - AlphaFold-Multimer parameters are downloaded into /workspace/params during build

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    bzip2 \
    aria2 \
    ffmpeg \
    procps \
  && rm -rf /var/lib/apt/lists/*

# Install micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_BANNER=1
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# Create conda environment from environment.yml
WORKDIR /workspace
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n germinal -f /tmp/environment.yml && micromamba clean -a -y

# Make the env the default for subsequent RUN/CMD
ENV PATH=/opt/conda/envs/germinal/bin:${PATH}
ENV PIP_NO_CACHE_DIR=1

# Install PyRosetta via conda (Rosetta Commons channel)
ARG PYROSETTA_CHANNEL=https://conda.rosettacommons.org
RUN micromamba install -y -n germinal -c "${PYROSETTA_CHANNEL}" pyrosetta && micromamba clean -a -y

# Install JAX (GPU or CPU), PyTorch (GPU or CPU), Torch Geometric, and additional deps
# You can build a CPU-only image with: 
#   docker build --build-arg JAX_CUDA=false --build-arg CUDA_SUFFIX=cpu \
#                --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu -t germinal:cpu .

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG CUDA_SUFFIX=cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${CUDA_SUFFIX}
ARG JAX_CUDA=true

RUN set -euxo pipefail; \
  python -m pip install --upgrade pip wheel setuptools; \
  if [ "${JAX_CUDA}" = "true" ]; then \
    python -m pip install "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
  else \
    python -m pip install "jax==0.5.3"; \
  fi; \
  python -m pip install --index-url "${TORCH_INDEX_URL}" \
    torch=="${TORCH_VERSION}" \
    torchvision=="${TORCHVISION_VERSION}" \
    torchaudio=="${TORCHAUDIO_VERSION}"; \
  if [ "${JAX_CUDA}" = "true" ]; then \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}%2B${CUDA_SUFFIX}.html"; \
  else \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"; \
  fi; \
  python -m pip install "torch_geometric==2.6.*" -f "${PYG_URL}"; \
  python -m pip install \
    chex dm-haiku==0.0.13 dm-tree joblib ml-collections immutabledict optax \
    pandas matplotlib numpy biopython scipy seaborn tqdm py3dmol \
    iglm chai-lab==0.6.1 torchtyping==0.1.5; \
  true

# Install colabfold ignoring dependency constraints (intentional)
RUN python -m pip install --no-deps colabfold==1.5.5

# Install ColabDesign (editable)
COPY colabdesign /workspace/colabdesign
RUN python -m pip install -e /workspace/colabdesign

# Copy repo and install Germinal (editable)
COPY . /workspace
RUN python -m pip install -e .

# Create mount points for data/outputs
RUN mkdir -p /workspace/params /workspace/pdbs /workspace/runs

# Download AlphaFold-Multimer parameters into the image
RUN set -euxo pipefail; \
  cd /workspace/params; \
  aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar; \
  tar -xf alphafold_params_2022-12-06.tar -C /workspace/params; \
  rm -f alphafold_params_2022-12-06.tar; \
  chmod -R a+rX /workspace/params

# Useful environment variables
ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    HYDRA_FULL_ERROR=1

# Add entrypoint that raises ulimit and execs in env
COPY docker-entrypoint.sh /usr/local/bin/germinal-entrypoint.sh
RUN chmod +x /usr/local/bin/germinal-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/germinal-entrypoint.sh"]
CMD ["bash"]


