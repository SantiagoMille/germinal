# Germinal: Efficient generation of epitope-targeted de novo antibodies

<p align="center">
  <img src="assets/germinal.png" alt="Germinal Banner"/>
</p>


Germinal is a pipeline for designing de novo antibodies against specified epitopes on target proteins. The pipeline follows a 3-step process: hallucination based on ColabDesign, selective sequence redesign with AbMPNN, and cofolding with a structure prediction model. Germinal is capable of designing both nanobodies and scFvs against user-specified residues on target proteins. 

We describe Germinal in the preprint: ["Efficient generation of epitope-targeted de novo antibodies with Germinal"](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1)

**⚠️ We are still actively working on code improvements [See our recommendations/tips](#tips-for-design)**. The Protenix and AbLang integrations are under active development — if you run into any issues please [open a GitHub issue](https://github.com/SantiagoMille/germinal/issues).

> **Last user-validated commit:** [`2c0a13b`](https://github.com/SantiagoMille/germinal/commit/2c0a13b76833b6463cb59c571cfeadf17fd710c1) (PR #61 "Fix tokenizer for IgLM"). Commits after this point have been runtime- and review-validated separately on branch `fix/post-pr67-review`.

## Important changes since `2c0a13b`

The commits below are bundled in PRs #55, #64, #67, #68, #69, #70 (merged) plus
the in-flight branch `fix/post-pr67-review` (~22 follow-up fixes). User-visible
behavior changes you should know about:

### New features / config knobs
- **Protenix structure model** (PR #55): `structure_model: "protenix"` is now
  fully supported alongside `af3` and `chai`. Set `protenix_conda_env` and
  `protenix_model_name` in your run config.
- **AbLang language model** (PR #55, #64, #68): `ablm_model: "ablang"` is now
  available alongside `iglm`. Method controlled by `ablm_method` (`"pll"`
  default, or `"unmasked"`).
- **MSA mode "target"** is the default for AF3/Protenix runs — MSA generated
  only for the target. Use `msa_mode: "colabfold"` for a real binder MSA.
- **Binder MSA caching** (`cache_binder_msa: true`) — reuses the first binder's
  ColabFold MSA across all subsequent designs by rewriting only the query
  line. Requires `msa_mode: "colabfold"`; raises ValueError otherwise.
- **AF3/Protenix sample selection** (`af3_structure_select_mode`): pick the
  `"best"` (highest ranking_score) or `"worst"` AF3/Protenix sample.
- **Multi-relax ensemble** (`multi_relax: true`, `n_relax: 5`,
  `relax_score_mode: "average"|"best"`): post-prediction PyRosetta relax can
  now spawn N parallel relaxes and aggregate their interface scores.
- **VL-first scFv** (PR #67): scFv runs with `vh_first: false` now correctly
  identify H-CDR3 from the last CDR (was previously mis-sliced into VL).

### Behavior changes / safer defaults
- **`evaluate_filters` fail-loud on None metrics**: any filter whose metric is
  `None` (e.g. Protenix without `full_data`) now FAILS the filter loudly
  instead of silently passing. If you relied on the old silent-pass, either
  fix the upstream metric or remove the filter.
- **AbMPNN pipeline failures abort the run**: previously, an AbMPNN worker
  crash returned `([], False)` and the trajectory loop continued silently.
  Now a `[ABMPNN PIPELINE FATAL]` message is printed and the job exits so
  the underlying issue can be diagnosed.
- **Parallel-relax worker exceptions are surfaced**: `_relax_worker` now
  writes a traceback to `{pdb}.err`; `pr_relax_parallel` reads and prints
  it inside a `[RELAX ERROR]` block.
- **Unknown `grad_merge_method` defaults to pcgrad** with a `[CONFIG WARNING]`
  (was: silent drop of the AbLang/IgLM gradient).
- **`get_grad_mlm` removed** from `colabdesign.ablang` (CUDA RNG state leak;
  callers should use `method="pll"` or `method="unmasked"`).

### Bug fixes (silent correctness issues)
- **Chai cdr3_idx now uses the correct H3 residue** (PR #67 + follow-up): both
  the CDR-position selection (PR #67) and the 0→1-indexed PDB residue number
  conversion are now correct. scFv × Chai × VL-first × hotspot runs were
  previously pinned one residue away from H3.
- **Protenix binder MSA**: `_get_or_generate_msas` now generates a real binder
  MSA when `msa_mode in {"local", "colabfold"}` (was target-only regardless).
- **`_unwrap` recursion**: handles arbitrary depth of single-element list
  wrapping in Protenix's tensor serialization (was: `[[scalar]]` → `[scalar]`).
- **`interface_cdrs` zero-division guard**: empty interface no longer kills
  the trajectory after structure prediction completed.
- **`idx_from_ranges` empty input**: returns `[]` instead of IndexError.
- **`lm_ll = -100` sentinel** for unknown `ablm_model` (was `-1`, easily
  confused with a real bad pseudolikelihood).
- **Chai tmp_dir mkdir(exist_ok=True)** to survive hash collisions.
- **pDockQ2 chain-keying** (drop buggy per-chain aggregation, use ipsae's
  scalar `pdockq2`/`LIS` directly).
- **AbLang+IgLM `ablm_method` AttributeError** (PR #69 follow-up): defensive
  `getattr(..., None)` for models that don't set this attribute.

### Config additions you should review
All five tracked configs in `configs/run/` now expose:
- `multi_relax`, `n_relax`, `relax_score_mode` (only `scfv_pdl1.yaml` had
  these before)
- `cache_binder_msa` (was missing everywhere)
- `ablm_model` is now present in `vhh_il3.yaml` (was missing → KeyError)

## Contents

<!-- TOC -->

- [Setup](#setup)
   * [Requirements](#requirements)
   * [Installation](#installation)
   * [Docker](#docker)
- [Usage](#usage)
   * [Quick Start](#quick-start)
      + [Configuration Structure](#configuration-structure)
   * [Basic Usage](#basic-usage)
   * [CLI Overrides](#cli-overrides)
   * [Target Configuration](#target-configuration)
   * [Filters Configuration](#filters-configuration)
   * [AF3 Configuration](#af3)
   * [Protenix Configuration](#protenix)
   * [AbLang Configuration](#ablang)
   * [Structure Score Selection Mode](#score-selection)
- [Output Format](#output-format)
- [Tips for Design](#tips-for-design)
- [Designing against PD-L1 and IL3](#design-against-pdl1-il3)
- [Troubleshooting](#troubleshooting)
- [Bugfix Changelog](#bugfix-changelog)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Community Acknowledgments](#community-acknowledgments)

<!-- TOC -->

<!-- TOC --><a name="setup"></a>
## Setup

<!-- TOC --><a name="requirements"></a>
### Requirements

**Prerequisites:**
- [PyRosetta](https://www.pyrosetta.org/) (academic license required)
- [ColabDesign/AlphaFold-Multimer parameters](https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar) (click link for download or see below for cli)
- [AlphaFold3 parameters](https://github.com/google-deepmind/alphafold3) (optional)
- JAX with GPU support

**System Requirements:**
- **GPU**: NVIDIA GPU with CUDA support
- **Memory**: 40GB+ VRAM*
- **Storage (recommended)**: 50GB+ space for results

> *The pipeline has been tested on: A100 40GB, H100 40GB MIG, L40S 48GB, A100 80GB, and H100 80GB.
> These runs tested a 130 amino acid target with a 131 amino acid nanobody. For larger runs, we recommend 60GB+ VRAM.

<!-- TOC --><a name="installation"></a>
### Installation

1. Ensure you have an NVIDIA GPU with a recent driver (recommended CUDA 12+). You can verify with:
   ```bash
   nvidia-smi
   ```
2. Install Miniconda or Anaconda if not already available.

3. Follow the **instructions** in `environment_setup.md`

4. Copy AlphaFold-Multimer parameters to `params/` and untar them. 
   Alternatively, you can run the following lines inside `params/` to download and untar:
   ```bash
   aria2c -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
   tar -xf alphafold_params_2022-12-06.tar -C .
   ```

5. Activate the environment:
   ```bash
   conda activate germinal
   ```

6. (Optional) Run validation at any time to ensure all packages have installed correctly:
   ```bash
   python validate_install.py
   ```

Notes:
- AlphaFold-Multimer and AlphaFold3 parameters are large and must be downloaded manually.

<!-- TOC --><a name=docker"></a>
### Docker
Germinal can be run using Docker:

```bash
docker build -t germinal .
docker run -it --rm --gpus all \
  -v "$PWD/results:/workspace/results" \
  -v "$PWD/pdbs:/workspace/pdbs" \
  germinal bash
```

and Singularity (shown)/Apptainer:
```bash
mkdir -p results
singularity pull germinal.sif docker://jwang003/germinal:latest
singularity shell --nv \
  --bind "$PWD/results:/workspace/results" \
  --bind "$PWD/pdbs:/workspace/pdbs" \
  --pwd /workspace \
  germinal.sif
```
> **Note:** Pulling may hang on `Creating SIF file...` If so, check if the command is done with `singularity exec germinal.sif python -c "print('ok')"`

Volumes are mounted to save generated input complexes and results from sampling.

Once inside the container you can test:
```bash
python run_germinal.py
```

<!-- TOC --><a name="usage"></a>
## Usage

<!-- TOC --><a name="quick-start"></a>
### Quick Start

The main entry point to the pipeline is `run_germinal.py`. Germinal uses [Hydra](https://hydra.cc/) for orchestrating different configurations. An example main configuration file is located in `configs/config.yaml`. This yaml file contains high level run parameters as well as pointers to more granular configuration settings.

These detailed options are stored in four main settings files:

 - **Main run settings**: `configs/run/vhh.yaml`
 - **Target settings**: `configs/target/[your_target].yaml`
 - **Post-hallucination (initial) filters**: `configs/filter/initial/[vhh/scfv].yaml`
 - **Final filters**: `configs/filters/final/[vhh/scfv].yaml`

<!-- TOC --><a name="configuration-structure"></a>
#### Configuration Structure (example)

```
configs/
├── config.yaml              # Main configuration yaml
├── run/                     # Main run settings
│   ├── vhh.yaml             # Example VHH (nanobody) settings
│   └── ...            		 # Other settings
├── target/                  # Target protein configurations
│   ├── pdl1.yaml            # PDL1 target example
│   └── ...             	 # other targets
└── filter/                  # Filter configurations
    ├── initial/
    │   ├── vhh.yaml     	 # Post-hallucination (initial) filters
    │   └── ...
    └── final/
        ├── vhh.yaml     	 # Final acceptance filters
        └── ...        
``` 

To design nanobodies targeting PD-L1 using default configs (with `chai` as the default structure predictor):

```bash
python run_germinal.py
```

To design scFvs targeting PD-L1 using default configs:

```bash
python run_germinal.py run=scfv filter/initial=scfv filter/final=scfv
```
> **Note:** Default configs are not meant to work well out of the box but rather be a set of reasonable default parameters that we used as a starting point for parameter exploration and sweep experiments.

If you wish to change the configuration of runs, you can:

 - create an entirely new config yaml
 - swap one of the four main settings files
 - pass specific overrides

<!-- TOC --><a name="basic-usage"></a>
### Basic Usage

**Run with defaults:**
```bash
python run_germinal.py
```

**Switch to a different run config (e.g., new_config):**
```bash
python run_germinal.py run=new_config
```

**Use different target:**
```bash
python run_germinal.py target=my_target
```

**Use a different config file with Hydra:**
```bash
python run_germinal.py --config_name new_config.yaml
```

**Use different filters:**
```bash
python run_germinal.py filter/initial=new_init_filter filter/final=new_final_filter
```

<!-- TOC --><a name="cli-overrides"></a>
### CLI Overrides

Hydra provides powerful CLI override capabilities. You can override any parameter in any configuration file.

> **!NOTE** Settings in `configs/run/` folder use the global namespace and do not need a `run` prefix before overriding. See example below.

**Basic parameter overrides:**
```bash
# Override trajectory limits
python run_germinal.py max_trajectories=100 max_passing_designs=50

# Override experiment settings
python run_germinal.py experiment_name=my_experiment run_config=test_run

# Override loss weights. Note: no run prefix since run settings are global
python run_germinal.py weights_plddt=1.5 weights_iptm=0.8 
```

**Filter threshold overrides:**
```bash
# Make initial filters less stringent
python run_germinal.py filter.initial.clashes.value=2

# Adjust final filter thresholds
python run_germinal.py filter.final.external_plddt.value=0.9 filter.final.external_iptm.value=0.8

# Change filter operators
python run_germinal.py filter.final.sc_rmsd.operator='<=' filter.final.sc_rmsd.value=5.0
```

**Target configuration overrides:**
```bash
# Change target hotspots
python run_germinal.py target.target_hotspots=\'A26,A30,A36,A44\'

# Use different PDB file
python run_germinal.py target.target_pdb_path=\'pdbs/my_target.pdb\' target.target_name=\'my_target\'
```

**Complex multi-parameter overrides:**
```bash
# Complete scFv run with custom settings
python run_germinal.py \
  run=scfv \
  target=pdl1 \
  max_trajectories=500 \
  experiment_name=\'scfv_pdl1_test\' \
  target.target_hotspots=\'A37,A39,A41\' \
  filter.final.external_plddt.value=0.85 \
  weights_iptm=1.0
```


<!-- TOC --><a name="target-configuration"></a>
### Target Configuration

For each new target, you will need to define a target settings yaml file which contains all relevant information about the target protin. Here is an example:

```yaml
target_name: "pdl1"
target_pdb_path: "pdbs/pdl1.pdb"
target_chain: "A"
binder_chain: "B"
target_hotspots: "25,26,39,41"
dimer: false  # support coming soon!
length: 133
```

<!-- TOC --><a name="filters-configuration"></a>
### Filters Configuration

There are two sets of filters: post-hallucination (initial) filters and final filters. The post-hallucination filters are applied after the hallucination step to determine which sequences to proceed to the redesign step. This filter set is a subset of the final filters, which is applied at the end of the pipeline to determine passing antibody sequences. Here is an example of the post-hallucination filters:
```yaml
clashes:
  value: 1
  operator: '<'

sc_rmsd:
  value: 7.0
  operator: '<'

binder_near_hotspot:
  value: true
  operator: '=='
```

**Multi-relax ensemble (`multi_relax`):** by default Germinal runs a single PyRosetta FastRelax per structure during final filters. Setting `multi_relax: true` runs `n_relax` relaxations in parallel with different random seeds. `relax_score_mode` controls how the ensemble result is reported: `"average"` (default) averages numeric metrics across all runs; `"best"` returns metrics from the single lowest-energy run only. In both modes the structure with the lowest `binder_score` (most negative REU) is saved.

```yaml
multi_relax: true
n_relax: 5
relax_score_mode: "average"  # or "best"
```

> **⚠️ `multi_relax` is under active development.** If you encounter any issues, please [open a GitHub issue](https://github.com/SantiagoMille/germinal/issues).

**AbLang sequence score (`lm_ll`):** the `lm_ll` column in `designs.csv` is the AbLang pseudo-log-likelihood of the final sequence — each residue is masked once and scored against full bidirectional context. Higher values indicate more natural antibody sequences. Uses AbLang1 for VHH and AbLang2 for scFv. Computed automatically for all accepted designs; useful as a post-hoc ranking criterion.

<!-- TOC --><a name="af3"></a>
### AF3 Configuration

To run AF3 in Singularity, we use 5 fields in the configuration, which are described below:

```yaml
af3_repo_path: "/path/to/alphafold3/repo"
af3_sif_path: "/path/to/alphafold3/sif"
af3_model_dir: "/path/to/alphafold3/weights"
af3_db_dir: "/path/to/alphafold3/databases"
msa_db_dir: "/path/to/colabfold/databases"
```

<!-- TOC --><a name="protenix"></a>
### Protenix Configuration

[Protenix](https://github.com/bytedance/Protenix) is an open-source reimplementation of AlphaFold 3 by ByteDance. It can be used as an alternative structure prediction backend alongside AF3 and Chai. **AF3 remains the recommended backend** — all published filter thresholds are calibrated against AF3 and Protenix has not been independently validated. Use Protenix when AF3 is not available (e.g. no Singularity/license access), but treat results as experimental.

Protenix has dependencies that conflict with the main `germinal` environment, so it must be installed in a **separate conda environment**. The pipeline invokes it via `conda run -n <env>`.

```bash
conda create --name protenix python=3.10 && conda activate protenix && pip install protenix
```

Before running, download the model weights by following the [Protenix model download instructions](https://github.com/bytedance/Protenix?tab=readme-ov-file#model-weights). The `protenix_model_name` field must match the downloaded checkpoint name exactly.

To use Protenix, set `structure_model: "protenix"` in your run config (or pass `structure_model=protenix` on the CLI) and configure the two required fields:

```yaml
protenix_conda_env: "protenix"                       # conda env with protenix installed
protenix_model_name: "protenix_base_default_v1.0.0"  # model checkpoint name
```

Optional speed tuning parameters (can also be set via CLI): `protenix_use_msa` (default `true`), `protenix_samples` (default `5`), `protenix_cycles` (default `10`), `protenix_steps` (default `200`).

> **`protenix_use_msa`**: Setting this to `false` skips Protenix's built-in MSA search (~3 min per prediction). For de novo antibody design there are typically no real homologs, so this is often acceptable for speed. However, disabling MSA may reduce confidence score accuracy — use `false` for faster runs and `true` when confidence quality is the priority.

> **Known limitation**: When Protenix does not produce full PAE matrix output, interface metrics (`i_pae`, `i_plddt`) are unavailable and any filters on those metrics will be automatically passed. A warning is printed when this occurs.

> **⚠️ Protenix support is under active development.** If you encounter any issues, please [open a GitHub issue](https://github.com/SantiagoMille/germinal/issues).

<!-- TOC --><a name="ablang"></a>
### AbLang Configuration

Germinal uses an antibody language model (AbLang by default) to bias hallucination towards sequences with high naturalness. The language model gradient is mixed with the structural gradient at each step using the method specified by `grad_merge_method` (default: `"pcgrad"`).

Key config parameters:

```yaml
ablm_model: "ablang"    # "ablang" (default) or "iglm"
ablm_method: "pll"      # gradient method: "pll" (default), "mlm", or "unmasked"
ablm_scale: [0.1, 0.4, 0.4, 1.0]  # ramp schedule (see below)
ablm_temp: 0.6          # softmax temperature for sequence sampling
grad_merge_method: "pcgrad"  # how to combine AF2 and AbLang gradients: "pcgrad", "scale", or "mgda"
```

**Gradient methods:**
- `"pll"` (default): Salazar-style masked PLL — each position is masked once and scored; most principled but slower. Forward passes are chunked (default 8 for scFv, 32 for VHH) to bound GPU memory.
- `"mlm"`: random-subset MLM — masks ~15% of positions per step; fast and stochastic.
- `"unmasked"`: single forward pass cross-entropy; fastest but not true PLL.

**`ablm_scale` ramp:** controls language model influence at each design phase. Defined as `[v1, v2, v3, v4]`:
- Logits phase: ramps linearly from `v1` → `v2`
- Softmax phase: holds at `v3`
- Semigreedy phase: uses `v3`
- Best-sequence selection criterion: uses `v4` as the LM score weight

**Memory:** default chunk sizes are 8 (scFv / AbLang2) and 32 (VHH / AbLang1). If you encounter OOM errors, reduce with `pll_chunk_size=4` or lower, or set `ablm_method: "mlm"` for a single-pass alternative. Recommended: set `export XLA_PYTHON_CLIENT_PREALLOCATE=false` and `export XLA_CLIENT_MEM_FRACTION=0.5`.

> **⚠️ AbLang integration is under active development.** If you encounter any issues, please [open a GitHub issue](https://github.com/SantiagoMille/germinal/issues).

<!-- TOC --><a name="score-selection"></a>
### Structure Score Selection Mode

When using AF3 or Protenix, the structure predictor generates multiple samples per design. The `af3_structure_select_mode` setting controls which sample is used for scoring:

```yaml
af3_structure_select_mode: "best"   # or "worst"
```

- **`"best"`** (default): Use the top-ranked sample by ranking score. This is the standard AF3 behavior.
- **`"worst"`**: Use the lowest-ranked sample by ranking score. This acts as a conservative filter -- only designs that score well even on their worst prediction will pass the pipeline thresholds.

This can also be set from the command line:

```bash
python run_germinal.py af3_structure_select_mode=worst
```

> **Note:** This setting applies to both the AF3 and Protenix backends. It does not apply to Chai.

<!-- TOC --><a name="multichain"></a>
### Multi-chain target input

To design against multi-chain targets, it is necessary to create a `target` YAML with multiple chains. See `configs/target/insulin.yaml` for more details. Make sure that:
- the binder chain is always the last chain (i.e. `target_chain: "A,B"` and `binder_chain: "C"`) in the target YAML file.
- the supplied target and binder PDBs have the correct chain naming (i.e. target PDB has chain A, B, ... & binder PDB has only one chain A). The pipeline generates a complex PDB that combined all chains in the right order using both target and binder PDBs. This process sets the binder as the last chain. 
- if using Chai's contact restraint option, by default use a `hotspot_residue` in chain A in the target YAML or **modify `chai.restraints` accordingly** to ensure the residue corresponds to the correct chain.
- *NOTE: this feature should be considered experimental as is still under development.*

<!-- TOC --><a name="output-format"></a>
## Output Format

Germinal generates organized output directories:

```
runs/your_target_nb_20240101_120000/
├── final_config.yaml           # Complete run configuration after overrides
├── trajectories/               # Results for trajectories which pass hallucination but fail the first set of filters
│   ├── structures/     
│   ├── plots/            
│   └── designs.csv      
├── redesign_candidates/        # Results for trajectories which are AbMPNN redesigned but fail the final set of filters
│   ├── structures/          
│   └── designs.csv           
├── accepted/                   # Antibodies that pass all filters
│   ├── structures/          
│   └── designs.csv           
├── all_trajectories.csv        # Main CSV containing designs in all three folders above
└── failure_counts.csv          # CSV logging # trajectories failing each step of hallucination
```

**Key Output Files:**
- `accepted/structures/*.pdb` - Final antibody-antigen structure for passing antibody designs.
- `all_trajectories.csv` - Complete list of designs that passed hallucination, their *in silico* metrics, which stage they reached, and the pdb path to the designed structure.

<!-- TOC --><a name="tips-for-design"></a>
## Important Notes and Tips for Design

Hallucination is inherently expensive. Designing against a 130 residue target takes anywhere from 2-8 minutes for a nanobody design iteration on an H100 80GB GPU, depending on which stage the designed sequence reaches. For 40GB GPUs or scFvs, this number is around 50% larger.

During sampling, we typically run antibody generation until there are around 1,000 passing designs against the specified target and observe a success rate of around ~1 per GPU hour. Of those, we typically select the top 40-50 sequences for experimental testing based on a combination of *in silico* metrics described in the preprint. While *in silico* success rates vary wildly across targets, we estimate that 200-400 H100 80GB GPU hours of sampling are typically enough to generate ~200 successful designs and some functional antibodies. 

Please consider that:

- We strongly recommend use of [AF3](https://github.com/google-deepmind/alphafold3) for design filtering as done in the paper, as **filters are only calibrated for AF3 confidence metrics**. We are actively working to add Chai calibrated thresholds for commercial users. Until then, running Germinal with `structure_model: "chai"` and NOT `structure_model: "af3"` should be considered experimental and may have lower passing rates. [Protenix](https://github.com/bytedance/Protenix) (`structure_model: "protenix"`) is also supported as a third structure prediction backend. Since Protenix is an open-source reimplementation of AF3, its confidence metrics may be similar in nature, but filter thresholds have not been independently validated against Protenix outputs and this option should be considered experimental. See the [Protenix Configuration](#protenix) section for setup instructions. Note that the current AF3 implementation assumes singularity for containerization. We are currently working on a Docker compatible wrapper, but if you need to run AF3 with Docker in the meantime, `_run_af3` in `germinal/filters/af3.py` holds the Singularity wrapper which should only need slight tweaks to run with Docker. More details on configuring AF3 are [here](#af3).
- While nanobody design is fully functional and validated experimentally, the configs and filters for scFvs remain preliminary; this functionality should therefore still be regarded as experimental.
- As recommended in the preprint, we suggest performing a small parameter sweep before launching full sampling runs. This is especially important when working with a new target or selecting a new epitope. In `configs/run/vhh_pdl1.yaml` and `configs/run/vhh_il3.yaml`, we provide the parameters that we used for PD-L1 and IL3 nanobody generations in the pre-print. We also include the filters used for these runs under `configs/filter/initial/` and `configs/filter/final/`. In `configs/run/vhh.yaml` and `configs/run/scfv.yaml` we provide a set of reasonable default parameters that we used as a starting point for parameter exploration and sweep experiments (see below **Important Notes and Tips for Design** for more details). One important distinction is that the structure model in the default nanobody configuration is `chai` instead of `af3` in order to allow users to run the pipeline with no additional setup. Note that final sampling runs in the preprint all used slightly modified parameters. Parameters can be configured from the command line. For example, you can set `weights_beta` and `weights_plddt` with the following command:

```bash
python run_germinal.py weights_beta=0.3 weights_plddt=1.0
```
- Support for multi-chain target inputs has been added, yet it **should still be considered experimental**. An example config file `configs/run/multichain_exmpl_insulin.yaml`, as well as an target file `configs/target/insulin.yaml`, can be used as starting point. Make sure that: 1) all chains in PDB have the right chain IDs (ideally A, B, C, etc.) and match what the target YAML file used. 2) the binder chain should always be the last chain in the target config YAML file (e.g. "B" for 1 chain target, "C" for 2 chain target, "D" for 3 chain target, etc.).
- Now it is possible to add contact restraints for Chai. This could help improve confidence. See `germinal/filters/chai.restraints`.
- Binder-specific pLDDT score (`plddt_binder`) has been added and can now be used to filter designs.
- pDockQ has been deprecated and no longer used. We still keep pDockQ2.


**Tweaking Parameters:**

Optimal design parameters are different for each target and antibody type! If you are experiencing low success rates, we recommend tweaking interface confidence weights (ipTM / iPAE), structure-based weights (helix, beta, framework loss), or the IgLM weights defined in `ablm_scale`.  In particular we recommend playing around with:

```python
weights_plddt: 1.0
weights_pae_inter: 0.5
weights_iptm: 0.7
weights_helix: 0.1
weights_beta: 0.1
framework_contact_offset: 1
```

`ablm_scale` is a key parameter that controls the influence of the antibody language model (AbLang/IgLM) during different stages of the design process. See the [AbLang Configuration](#ablang) section for a full description of the ramp schedule and gradient methods.

Filters are also easily changeable in the filters configurations. To add or remove filters from the initial and final filtering rounds, simply create a new filter with the same name as the intended metric and specify the threshold value and the operator (<, >, =, etc).

Finally, using omit_AAs - e.g. `omit_AAs: "C,A"` in the yaml or `omit_AAs="'C,A'"` in the command line (note the double quotation for hydra) - allows one to omit any amino acid from appearing in the CDRs, opposed to all of the protein.

An example of a param sweep could be:

```bash
python run_germinal.py weights_beta=0.1 weights_helix=0.1 weights_plddt=1.0 experiment_name=beta01-helix01-plddt1

python run_germinal.py weights_beta=0.3 weights_helix=0.2 weights_plddt=1.5 experiment_name=beta03-helix02-plddt1.5
...
```

More tips coming soon!

<!-- TOC --><a name="design-against-pdl1-il3"></a>
## Designing against PD-L1 and IL3

PD-L1 VHH preprint config:
```bash
python -u run_germinal.py run=vhh_pdl1 experiment_name=pdl1_vhh filter/initial=vhh_pdl1 filter/final=vhh_pdl1 target=pdl1
```

IL3 VHH preprint config:
```bash
python -u run_germinal.py run=vhh_il3 experiment_name=il3_vhh filter/initial=vhh_il3 filter/final=vhh_il3 target=il3
```

PD-L1 scFV (not experimentally validated yet) config:
```bash
python -u run_germinal.py run=scfv_pdl1 experiment_name=pdl1_scfv filter/initial=scfv_pdl1 filter/final=scfv_pdl1 target=pdl1
```

<!-- TOC --><a name="troubleshooting"></a>
## Troubleshooting
- We have occasionally observed OOM errors when using AbLang 1-heavy to design VHHs. If you are experiencing this error, set `export XLA_PYTHON_CLIENT_PREALLOCATE=false` and `export XLA_CLIENT_MEM_FRACTION=0.5`.
- OOM errors during the AbLang PLL gradient step: set `export XLA_PYTHON_CLIENT_PREALLOCATE=false` and `export XLA_CLIENT_MEM_FRACTION=0.5` to limit JAX pre-allocation. Default chunk sizes are 8 (scFv) and 32 (VHH); reduce with `pll_chunk_size=4` if needed. Alternatively, switch to the single-pass MLM method with `ablm_method=mlm`.

<!-- TOC --><a name="bugfix-changelog"></a>
## Bugfix Changelog

- 9/25/25: Import fix for local colabdesign module ([commit 8b5b655](https://github.com/SantiagoMille/germinal/commit/8b5b655), [pr #8](https://github.com/SantiagoMille/germinal/pull/8)) 
- 9/25/25: A metric meant for tracking purposes `external_i_pae` was erroneously set to be used as a filter ([commit 49be2e9](https://github.com/SantiagoMille/germinal/commit/49be2e9), [issue #7](https://github.com/SantiagoMille/germinal/issues/7))
- 9/26/25: Resolved an error which caused passing runs to crash at the final stage due to a misnamed variable ([commit 9292e1e](https://github.com/SantiagoMille/germinal/commit/9292e1e), [issue #11](https://github.com/SantiagoMille/germinal/issues/11))
- 9/28/25: Resolved an error in throwing exception for AF3 calls + added containerization support ([commit e4ca63a](https://github.com/SantiagoMille/germinal/commit/e4ca63a), [raised in pr #12](https://github.com/SantiagoMille/germinal/pull/12))
- 10/1/25: Resolved a bug where trajectory sequence and structure path information was not updated after AbMPNN redesign. True sequence / structures can still be found in the pdb files in the `structures/` folders ([commit b45136c](https://github.com/SantiagoMille/germinal/commit/b45136c))
- 4/21/26: AbLang gradient refactor — chunked PLL to bound GPU memory for scFv, `ablm_method` config now correctly applied during hallucination, `run_germinal.py` unpack crash at filter stage fixed ([pr #68](https://github.com/SantiagoMille/germinal/pull/68))

<!-- TOC --><a name="citation"></a>
## Citation

If you use Germinal in your research, please cite:

```bibtex
@article{mille-fragoso_efficient_2025,
	title = {Efficient generation of epitope-targeted de novo antibodies with Germinal},
   author = {Mille-Fragoso, Luis Santiago and Wang, John N. and Driscoll, Claudia L. and Dai, Haoyu and Widatalla, Talal M. and Zhang, Xiaowe and Hie, Brian L. and Gao, Xiaojing J.},
	url = {https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1},
	doi = {10.1101/2025.09.19.677421},
	publisher = {bioRxiv},
	year = {2025},
}
```

<!-- TOC --><a name="acknowledgments"></a>
## Acknowledgments

Germinal builds upon the foundational work of previous hallucination-based protein design pipelines such as ColabDesign and BindCraft and this codebase incorporates code from both repositories. We are grateful to the developers of these tools for making them available to the research community. 

**Related Work:**
If you use components of this pipeline, please also cite the underlying methods:

- **ColabDesign**: [https://github.com/sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- **IgLM**: [https://github.com/Graylab/IgLM](https://github.com/Graylab/IgLM)
- **Chai-1**: [https://github.com/chaidiscovery/chai-lab](https://github.com/chaidiscovery/chai-lab)
- **AlphaFold3**: [https://github.com/google-deepmind/alphafold3](https://github.com/google-deepmind/alphafold3)
- **Protenix**: [https://github.com/bytedance/Protenix](https://github.com/bytedance/Protenix)
- **AbMPNN**: [Dreyer, F. A., Cutting, D., Schneider, C., Kenlay, H. & Deane, C. M. Inverse folding for
antibody sequence design using deep learning. (2023).](https://www.biorxiv.org/content/10.1101/2025.05.09.653228v1.full.pdf)
- **PyRosetta**: [https://www.pyrosetta.org/](https://www.pyrosetta.org/)

<!-- TOC --><a name="community-acknowledgments"></a>
## Community Acknowledgments

- [@cytokineking](https://github.com/cytokineking) - for helping identify issues with initial versions of the codebase
- [@shindo687](https://github.com/shindo687) - for helping ensure proper hyperlinks to supporting papers in the README
- [@azam-huss](https://github.com/azam-huss) - for helping consolidate the config arguments

## License

This repository is licensed under the [Apache License 2.0](LICENSE).

### External Dependencies

Some components require separate licenses that are not included in this repository:

- **IgLM**: Provided under a non-commercial academic license from Johns Hopkins University.  
  See their documentation for details.  
- **PyRosetta**: Provided by the Rosetta Commons and University of Washington under a non-commercial, non-profit license.  
  PyRosetta cannot be redistributed and must be obtained separately.  
  Commercial use requires a separate license. See [https://www.pyrosetta.org](https://www.pyrosetta.org).
