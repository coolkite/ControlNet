#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 30 minutes
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=2080ti # Request access to an m40 GPU
#SBATCH --mem=30G # Request 30GB of memory
#SBATCH --array=0-99 # Set the array size to 100
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Set the input and output data paths
input_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/data/03001627_pointbert_100_depth"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/data/03001627_pointbert_100_controlnet_final"

# Array of prompts
prompts=("a photo of a {}" "a rendering of a {}" "a cropped photo of the {}" "the photo of a {}" "a photo of a clean {}" "a photo of a dirty {}" "a dark photo of the {}" "a photo of my {}" "a photo of the cool {}" "a close-up photo of a {}" "a bright photo of the {}" "a cropped photo of a {}" "a photo of the {}" "a good photo of the {}" "a photo of one {}" "a close-up photo of the {}" "a rendition of the {}" "a photo of the clean {}" "a rendition of a {}" "a photo of a nice {}" "a good photo of a {}" "a photo of the nice {}" "a photo of the small {}" "a photo of the weird {}" "a photo of the large {}" "a photo of a cool {}" "a photo of a small {}")

# Array of seeds
# seeds=(1234 5678 9012 3456 7890)
seed_range=10000
views=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

# Get the list of model IDs
model_ids=($(ls -1 "${input_data_dir}" | grep -E '^[0-9]+_' | sort -u))

# Get the model ID for the current job
model_id="${model_ids[SLURM_ARRAY_TASK_ID]}"

echo "Processing depth maps for model ID: ${model_id}"

# Run the script with the specified arguments
python process_depth_maps.py \
    --model_id "$model_id" \
    --prompts "${prompts[@]}" \
    --seed_range $seed_range \
    --views "${views[@]}" \
    --num_samples_per_model 100 \
    --input_data_dir "$input_data_dir" \
    --output_data_dir "$output_data_dir" \
    --num_samples 1 \
    --image_resolution 512 \
    --ddim_steps 100 \
    --strength 1.2 \
    --scale 9.0 \
    --eta 0.0