#!/bin/bash
#SBATCH --job-name="mca-irony"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --mem=40G
#SBATCH -p kisski
#SBATCH -G A100:1

module load spack
spack load miniconda

python -m venv ~/envs/irony
source ~/envs/irony/bin/activate
pip install -r requirements.txt
hf auth login --token "$HF_TOKEN"
python run_inference.py

