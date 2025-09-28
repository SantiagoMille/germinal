# GPU runtime base with CUDA 12.4 + cuDNN on Ubuntu 22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bzip2 aria2 ffmpeg procps tini \
  && rm -rf /var/lib/apt/lists/*

ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV MAMBA_NO_BANNER=1
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | \
    tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

# create conda env
ENV CONDA_ENV=germinal
RUN micromamba create -y -n ${CONDA_ENV} python=3.10 && micromamba clean -a -y

ENV PATH=/opt/conda/envs/${CONDA_ENV}/bin:${PATH}
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg HYDRA_FULL_ERROR=1 PIP_NO_CACHE_DIR=1

WORKDIR /workspace

COPY setup.py README.md ./
COPY colabdesign /workspace/colabdesign

ARG TORCH_VERSION=2.6.0
ARG TORCHVISION_VERSION=0.21.0
ARG TORCHAUDIO_VERSION=2.6.0
ARG CUDA_SUFFIX=cu124                    # use 'cpu' for CPU-only builds
ARG JAX_CUDA=true                        # 'false' for CPU-only JAX
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${CUDA_SUFFIX}

# install pip packages
RUN set -euxo pipefail; \
  python -m pip install --upgrade pip wheel setuptools; \
  python -m pip install uv; \
  uv pip install --system pandas matplotlib numpy biopython scipy seaborn tqdm ffmpeg py3dmol \
    chex dm-haiku dm-tree joblib ml-collections immutabledict optax cvxopt mdtraj colabfold; \
  uv pip install --system -e /workspace/colabdesign; \
  uv pip install --system --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"; \ 
  uv pip install --system torchtyping==0.1.5; \
  if [ "${CUDA_SUFFIX}" = "cpu" ]; then \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"; \
  else \
    PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VERSION}%2B${CUDA_SUFFIX}.html"; \
  fi; \
  uv pip install --system "torch_geometric==2.6.*" -f "${PYG_URL}"; \
  uv pip install --system iglm chai-lab==0.6.1; \
  true

# pin jax version to 0.5.3
RUN set -euxo pipefail; \
  uv pip install --system jax==0.5.3; \
  uv pip install --system dm-haiku==0.0.13; \
  uv pip install --system hydra-core omegaconf; \
  if [ "${JAX_CUDA}" = "true" ]; then \
    uv pip install --system "jax[cuda12_pip]==0.5.3" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html; \
  fi; \
  true

# download pyrosetta
RUN set -euxo pipefail; \
  uv pip install --system pyrosetta-installer; \
  python -c "import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()"; \
  true

COPY . /workspace
RUN uv pip install --system -e .

# download multimer parameters
RUN set -euxo pipefail; \
  mkdir -p /workspace/params; \
  cd /workspace/params; \
  aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar; \
  tar -xf alphafold_params_2022-12-06.tar -C /workspace/params; \
  rm -f alphafold_params_2022-12-06.tar; \
  chmod -R a+rX /workspace/params

RUN mkdir -p /workspace/pdbs /workspace/results

CMD ["bash"]
