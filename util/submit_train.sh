#!/bin/bash
#SBATCH --job-name="mca-irony"
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --threads-per-core=1
#SBATCH --mem=15000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
spack load miniconda
source .env

python -m venv ~/envs/irony
source ~/envs/irony/bin/activate
pip install -r requirements.txt
hf auth login --token "$HF_TOKEN"
python run_inference.py