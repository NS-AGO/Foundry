#!/bin/bash

#SBATCH --job-name=foundry_experiment
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -e

# Activate the project environment
source .venv/bin/activate

# Run the Foundry experiment
uv run python main.py experiment=example/example_experiment