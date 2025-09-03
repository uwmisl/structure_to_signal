#!/bin/bash
#SBATCH --job-name=klone-container
#SBATCH --account=ml4ml
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time 72:00:00
#SBATCH --chdir=/gscratch/ml4ml/sidlak/structure_to_signal/
#SBATCH --output=finetune_out.txt
#SBATCH --error=finetune_err.txt

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(shuf -i 0-65535 -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo Lets go!

export PYTHONPATH="$PYTHONPATH:kmer-project" # path to python env
export HF_DATASETS_CACHE="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export HF_HOME="/mmfs1/gscratch/ml4ml/sidlak/.my_cache"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=0

srun accelerate launch --num_processes=1 --num_machines 1 --mixed_precision 'no' --dynamo_backend 'no' ./generation/gpt_regression_model.py --config_path="/gscratch/ml4ml/sidlak/structure_to_signal/generation/gpt_regression_config.json"