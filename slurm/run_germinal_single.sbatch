#!/bin/bash
#SBATCH --job-name=vhh_pdl1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

#SBATCH -p preemptible
#SBATCH --gpus=1
#SBATCH --output=jobs/vhh_pdl1-%j.out
#SBATCH --error=jobs/vhh_pdl1-%j.err
#SBATCH --requeue

python -u run_germinal.py run=vhh_pdl1 experiment_name=pdl1_vhh filter/initial=vhh_pdl1 filter/final=vhh_pdl1 target=pdl1

#Example for running with multi-chain target (e.g. insulin)
#python -u run_germinal.py run=/multichain_exmpl_insulin experiment_name=multichain filter/initial=vhh filter/final=vhh target=insulin

echo "finished"