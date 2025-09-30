#!/bin/bash
#SBATCH --job-name=example
#SBATCH --time=24:00:00
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=48G

#SBATCH -p preemptible
#SBATCH --gpus=1
#SBATCH --output=jobs/example-%j.out
#SBATCH --requeue

python -u run_germinal.py

echo "finished"