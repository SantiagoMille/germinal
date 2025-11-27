# Germinal: Efficient generation of epitope-targeted de novo antibodies

<p align="center">
  <img src="assets/germinal.png" alt="Germinal Banner"/>
</p>


Germinal is a pipeline for designing de novo antibodies against specified epitopes on target proteins. The pipeline follows a 3-step process: hallucination based on ColabDesign, selective sequence redesign with AbMPNN, and cofolding with a structure prediction model. Germinal is capable of designing both nanobodies and scFvs against user-specified residues on target proteins. 

We describe Germinal in the preprint: ["Efficient generation of epitope-targeted de novo antibodies with Germinal"](https://www.biorxiv.org/content/10.1101/2025.09.19.677421v1)

**⚠️ We are still actively working on code improvements [See our recommendations/tips](#tips-for-design)**. 

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

- We strongly recommend use of [AF3](https://github.com/google-deepmind/alphafold3) for design filtering as done in the paper, as **filters are only calibrated for AF3 confidence metrics**. We are actively working to add Chai calibrated thresholds for commercial users. Until then, running Germinal with `structure_model: "chai"` and NOT `structure_model: "af3"` should be considered experimental and may have lower passing rates. Note that the current AF3 implementation assumes singularity for containerization. We are currently working on a Docker compatible wrapper, but if you need to run AF3 with Docker in the meantime, `_run_af3` in `germinal/filters/af3.py` holds the Singularity wrapper which should only need slight tweaks to run with Docker. More details on configuring AF3 are [here](#af3).
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

`ablm_scale` is a key parameter that controls the influence of IgLM during different stages of the design process. `ablm_scale` is defined as a list of four scalar values: `[v_1,v_2,v_3,v_4]`. During the logits phase, ablm_scale increases linearly between v_1 and v_2. During the softmax phase, ablm_scale takes the value of v_3, and during the semi-greedy stage ablm_scale takes the value of v_4. 

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
- We have occassionally observed OOM errors when using AbLang 1-heavy to design VHHs. If you are experiencing this error, try lowering the amount of memory Jax preallocates to 0.5 with `export XLA_CLIENT_MEM_FRACTION=0.5` or ` XLA_CLIENT_MEM_FRACTION=0.5 python run_germinal.py [args]`.

<!-- TOC --><a name="bugfix-changelog"></a>
## Bugfix Changelog

- 9/25/25: Import fix for local colabdesign module ([commit 8b5b655](https://github.com/SantiagoMille/germinal/commit/8b5b655), [pr #8](https://github.com/SantiagoMille/germinal/pull/8)) 
- 9/25/25: A metric meant for tracking purposes `external_i_pae` was erroneously set to be used as a filter ([commit 49be2e9](https://github.com/SantiagoMille/germinal/commit/49be2e9), [issue #7](https://github.com/SantiagoMille/germinal/issues/7))
- 9/26/25: Resolved an error which caused passing runs to crash at the final stage due to a misnamed variable ([commit 9292e1e](https://github.com/SantiagoMille/germinal/commit/9292e1e), [issue #11](https://github.com/SantiagoMille/germinal/issues/11))
- 9/28/25: Resolved an error in throwing exception for AF3 calls + added containerization support ([commit e4ca63a](https://github.com/SantiagoMille/germinal/commit/e4ca63a), [raised in pr #12](https://github.com/SantiagoMille/germinal/pull/12))
- 10/1/25: Resolved a bug where trajectory sequence and structure path information was not updated after AbMPNN redesign. True sequence / structures can still be found in the pdb files in the `structures/` folders ([commit b45136c](https://github.com/SantiagoMille/germinal/commit/b45136c))

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
