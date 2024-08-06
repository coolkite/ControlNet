#!/bin/bash

#SBATCH -p gpu-preempt # Submit job to gpu-preempt partition
#SBATCH -t 23:00:00 # Set max job time for 23 hours
#SBATCH --ntasks=1 # Set the number of tasks to 1
#SBATCH --gpus-per-task=1 # Request 1 GPU per task
#SBATCH --constraint=m40 # Request access to an m40 GPU
#SBATCH --mem=15G # Request 15GB of memory
#SBATCH --array=0-1124 # Set the array size to cover all combinations
#SBATCH --output=logs/slurm-%A_%a.out # Specify the output log file
#SBATCH --error=logs/slurm-%A_%a.err # Specify the error log file

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Set the input and output data paths
input_data_dir="/work/pi_ekalogerakis_umass_edu/dmpetrov/data/token_geometry/shapenet_pointbert_1000/depth_images" #"/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/data/03001627_pointbert_100_depth"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/5test_debug_inloop_all"

# Define arrays for model_ids, seeds, views, and intervals
model_ids=(
    "03001627_1bec15f362b641ca7350b1b2f753f3a2"
    "03001627_1157d8d6995da5c0290d57214c8512a4"
    "03001627_1f5a2c231265aa9380b3cfbeccfb24d2"
    "03001627_2a417b5a946ff7eb2a3f8f484e6c5c4f"
    "03001627_8e2b44ec14701d057c2b071b8bda1b69"
)

seeds=(021904 170605 091222 270606 123115)
views=(0 4 8 12 16)
intervals=(18 19 20 21 22 23 24 25 50)

# Calculate indices for each parameter
model_id_index=$((SLURM_ARRAY_TASK_ID / (${#seeds[@]} * ${#views[@]} * ${#intervals[@]})))
seed_index=$((SLURM_ARRAY_TASK_ID / (${#views[@]} * ${#intervals[@]}) % ${#seeds[@]}))
view_index=$((SLURM_ARRAY_TASK_ID / ${#intervals[@]} % ${#views[@]}))
interval_index=$((SLURM_ARRAY_TASK_ID % ${#intervals[@]}))

# Get the values for the current job
model_id="${model_ids[model_id_index]}"
seed="${seeds[seed_index]}"
view="${views[view_index]}"
interval="${intervals[interval_index]}"

# Array of prompts
init_prompts=("a photo of a chair")
baseline_prompts=("a photo of a chair")
object="chair"
learned_token="<chair>"
learned_token_path="/work/pi_ekalogerakis_umass_edu/pgoyal/geometry_editing/trained_embeddings/text-inversion-model-main-matching-prompts-50-512-8"

ddim_steps=50

echo "Processing with model ID: ${model_id}, seed: ${seed}, view: ${view}, interval: ${interval}"

# Run the script with the specified arguments
python generate_images_lite_2_1.py \
    --model_id "$model_id" \
    --init_prompts "${init_prompts[@]}" \
    --baseline_prompts "${baseline_prompts[@]}" \
    --seed $seed \
    --view $view \
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
    --learned_token "$learned_token" \
    --learned_token_path "$learned_token_path"