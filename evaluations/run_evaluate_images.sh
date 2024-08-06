#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 23 hours
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=m40 # Request access to an m40 GPU
#SBATCH --mem=24G # Request 15GB of memory
#SBATCH --exclusive # Request exclusive access to the node
#SBATCH --array=0-7 # Set the array size to 8 (0-7) to match the number of intervals
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate dino-vit-feats-env

# Set the input and output data paths
input_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/ptest"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/ptest_output"
sam_checkpoint_path="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/segment-anything/sam_vit_h_4b8939.pth"

# Get the list of interval directories
interval_dirs=($(find "$input_data_dir" -mindepth 3 -maxdepth 3 -type d))

# Get the interval directory for the current job
interval_dir="${interval_dirs[SLURM_ARRAY_TASK_ID]}"

echo "Processing interval directory: $interval_dir"

# Run the script with the specified arguments
python evaluate_images.py \
    --input_data_dir "$interval_dir" \
    --output_data_dir "$output_data_dir" \
    --sam_checkpoint_path "$sam_checkpoint_path"