#!/bin/bash
#SBATCH --job-name="mca-irony"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --mem=40G
#SBATCH -G A100:1
#SBATCH --partition=kisski

source ~/envs/irony/bin/activate
pip install -r requirements.txt

set -a
source .env
set +a

export HF_HOME=/mnt/vast-kisski/projects/kisski-irony/hf_cache
export HF_HUB_OFFLINE=1
python run_inference.py

