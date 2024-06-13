#!/bin/bash

#SBATCH -p gpu-preempt
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=m40
#SBATCH --mem=50G
#SBATCH --array=0-245
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Set the prompt
prompt="a photo of a chair"
token=$(echo "$prompt" | tr ' ' '_')

# Define the list of seeds
seeds=(549110186 170605 021904 091222 270606)

# Get the list of image files in the assets folder
image_files=(partnet_42792_renders_green/partnet_42792/*.png)

# Get the number of image files
num_images=${#image_files[@]}

# Calculate the image index and seed index based on the array task ID
image_index=$((SLURM_ARRAY_TASK_ID / ${#seeds[@]}))
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))

# Get the image file path
image_file="${image_files[image_index]}"

# Extract the image name without extension
image_name=$(basename "$image_file" | cut -d. -f1)

# Create a folder for the image if it doesn't exist
mkdir -p "results/partnet_42792_green/$image_name"

# Get the seed value
seed=${seeds[seed_index]}

# Execute the Python script with the specific image, prompt, and seed
python no_gradio_depth2image_folder.py \
    --input_image "$image_file" \
    --prompt "$prompt" \
    --a_prompt "" \
    --n_prompt "" \
    --num_samples 1 \
    --image_resolution 500 \
    --detect_resolution 384 \
    --ddim_steps 50 \
    --strength 1.0 \
    --scale 9.0 \
    --seed $seed \
    --eta 0.0 \
    --output_folder "results/partnet_42792_green/$image_name"