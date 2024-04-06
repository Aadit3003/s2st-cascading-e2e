#!/bin/bash
#SBATCH --job-name=casc2
#SBATCH --output=bleu2.out
#SBATCH --error=bleu2.err
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=02:00:00

echo "LOADING THE ENVIRONMENT"
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate Speech
echo "Starting"

# Your job commands go here
python evaluation.py
echo "CASC DONE!!"
