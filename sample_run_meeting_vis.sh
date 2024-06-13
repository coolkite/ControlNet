#!/bin/bash

#SBATCH -p gpu-preempt             # Submit job to gpu-preempt partition
#SBATCH -t 04:00:00                # Set max job time for 4 hours
#SBATCH --ntasks=1                 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1          # Request 1 GPU per task
#SBATCH --constraint=m40           # Request access to an m40 GPU
#SBATCH --mem=50G                  # Request 50GB of memory
#SBATCH --array=0-5
#SBATCH --output=logs/slurm-%A_%a.out  # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err   # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Array of prompts
prompts=("a photo of a chair" "a painting of a chair" "a photo of a chair with green back" "a photo of a green back chair" "a photo of round leg chair" "a photo of square leg chair")

# Get the prompt based on the array index
prompt="${prompts[$SLURM_ARRAY_TASK_ID]}"
token=$(echo "$prompt" | tr ' ' '_')

# Execute the Python script with the specific prompt
srun python no_gradio_depth2image_periodic.py \
    --input_image assets/chair_white.png \
    --prompt "$prompt" \
    --a_prompt "" \
    --n_prompt "" \
    --num_samples 1 \
    --image_resolution 512 \
    --detect_resolution 384 \
    --ddim_steps 50 \
    --strength 1.0 \
    --scale 9.0 \
    --seed 170605 \
    --eta 0.0 \
    --interval 25