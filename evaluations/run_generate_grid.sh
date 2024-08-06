#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 23 hours
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=m40 # Request access to an m40 GPU
#SBATCH --mem=15G # Request 15GB of memory
#SBATCH --array=0 # Set the array size to cover all combinations
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control #dino-vit-feats-env

# Assign input arguments to variables
INPUT_DATA_DIR="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/sd_1_5_output"
OUTPUT_DATA_DIR="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/sd_1_5_output_grid"

# Create output directories if they don't exist
mkdir -p "$OUTPUT_DATA_DIR"

# Run the grid generation script for each mask method
for mask_method in a_original_depth_mask #c_baseline_dino_mask d_learned_dino_mask
do
    echo "Running grid generation script for $mask_method..."
    python generate_grid.py --input_dir "${INPUT_DATA_DIR}/${mask_method}" --output_dir "${OUTPUT_DATA_DIR}/${mask_method}"
done

echo "Processing complete. Grid images have been generated in $OUTPUT_DATA_DIR"