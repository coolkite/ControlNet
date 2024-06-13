#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 23 hours
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=m40 # Request access to an m40 GPU
#SBATCH --mem=30G # Request 30GB of memory
#SBATCH --array=0-3 # Set the array size to 100
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Set the input and output data paths
input_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/data/03001627_pointbert_100_depth"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data"

# Array of prompts
init_prompts=("a photo of a chair")
baseline_prompts=("a photo of a chair")
object="chair"
learned_token="<chair>"
learned_token_path="/project/pi_ekalogerakis_umass_edu/dmpetrov/data/textual_inversion_results/03001627_1006be65e7bc937e9141f9b58470d646/learned_embeds.safetensors"

# Array of seeds
# seeds=(1234 5678 9012 3456 7890)
seed_range=1
views=(0) # 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

# Get the list of model IDs
model_ids=($(ls -1 "${input_data_dir}" | grep -E '^[0-9]+_' | sort -u))

# Get the model ID for the current job
model_id="${model_ids[SLURM_ARRAY_TASK_ID]}"

ddim_steps=50
interval=25
echo "Processing depth maps for model ID: ${model_id}"

# Run the script with the specified arguments
python baseline_learned_control_k.py \
    --model_id "$model_id" \
    --init_prompts "${init_prompts[@]}" \
    --seed_range $seed_range \
    --views "${views[@]}" \
    --num_samples_per_model 1 \
    --input_data_dir "$input_data_dir" \
    --output_data_dir "$output_data_dir" \
    --num_samples 1 \
    --image_resolution 512 \
    --ddim_steps $ddim_steps \
    --strength 1.2 \
    --scale 9.0 \
    --eta 0.0 \
    --object "$object" \
    --interval $interval \
    --baseline_prompts "${baseline_prompts[@]}" \
    --learned_token "$learned_token" \
    --learned_token_path "$learned_token_path" 