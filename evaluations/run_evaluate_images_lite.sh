#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 23 hours
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=m40 # Request access to an m40 GPU
#SBATCH --mem=24G # Request 24GB of memory
#SBATCH --array=0-1499 # Set the array size to cover all combinations (5 shapes * 6 intervals * 3 views * 8 seeds)
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate dino-vit-feats-env

# Set the input and output data paths
input_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/sd_1_5"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/sd_1_5_output"
sam_checkpoint_path="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/segment-anything/sam_vit_h_4b8939.pth"

# Define arrays for shapes, intervals, views, and seeds
shapes=(
    "1bec15f362b641ca7350b1b2f753f3a2"
    "1157d8d6995da5c0290d57214c8512a4"
    "1f5a2c231265aa9380b3cfbeccfb24d2"
    "2a417b5a946ff7eb2a3f8f484e6c5c4f"
    "8e2b44ec14701d057c2b071b8bda1b69"
)
intervals=(0 18 19 20 21 22 23 24 25 50)
views=(0 4 8 12 16)
seeds=(0 021904 091222 270606 123115 170605)

# Calculate indices for each parameter
shape_index=$((SLURM_ARRAY_TASK_ID / (${#intervals[@]} * ${#views[@]} * ${#seeds[@]})))
interval_index=$((SLURM_ARRAY_TASK_ID / (${#views[@]} * ${#seeds[@]}) % ${#intervals[@]}))
view_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]} % ${#views[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

# Get the values for the current job
shape="${shapes[shape_index]}"
interval="${intervals[interval_index]}"
view="${views[view_index]}"
seed="${seeds[seed_index]}"

echo "Processing shape: $shape, interval: $interval, view: $view, seed: $seed"

# Run the script with the specified arguments
python evaluate_images_lite.py \
    --input_data_dir "$input_data_dir" \
    --output_data_dir "$output_data_dir" \
    --sam_checkpoint_path "$sam_checkpoint_path" \
    --shape "$shape" \
    --interval "$interval" \
    --view "$view" \
    --seed "$seed"