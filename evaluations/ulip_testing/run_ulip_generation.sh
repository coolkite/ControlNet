#!/bin/bash

#SBATCH -p gpu-preempt
#SBATCH -t 23:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=m40
#SBATCH --mem=15G
#SBATCH --array=0-1499
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

module load miniconda/22.11.1-1
conda activate control

input_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/rendered_images_depth"
output_data_dir="/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/ulip_testing/data/test_zoe"

model_ids=(
    "03001627-1157d8d6995da5c0290d57214c8512a4"
    "03001627-1bec15f362b641ca7350b1b2f753f3a2"
    "03001627-1f5a2c231265aa9380b3cfbeccfb24d2"
    "03001627-2a417b5a946ff7eb2a3f8f484e6c5c4f"
    "03001627-8e2b44ec14701d057c2b071b8bda1b69"
)

seeds=(0 021904 170605 091222 270606 123115)
views=(180 120 036 324 240)
intervals=(0 18 19 20 21 22 23 24 25 50)

model_id_index=$((SLURM_ARRAY_TASK_ID / (${#seeds[@]} * ${#views[@]} * ${#intervals[@]})))
seed_index=$((SLURM_ARRAY_TASK_ID / (${#views[@]} * ${#intervals[@]}) % ${#seeds[@]}))
view_index=$((SLURM_ARRAY_TASK_ID / ${#intervals[@]} % ${#views[@]}))
interval_index=$((SLURM_ARRAY_TASK_ID % ${#intervals[@]}))

model_id="${model_ids[model_id_index]}"
seed="${seeds[seed_index]}"
view="${views[view_index]}"
interval="${intervals[interval_index]}"

init_prompts=("a photo of a chair")
baseline_prompts=("a photo of a chair")
object="chair"
learned_token="<chair>"
learned_token_path="/work/pi_ekalogerakis_umass_edu/pgoyal/geometry_editing/trained_embeddings_prev/text-inversion-model-main-matching-prompts-50-512-4-6000-runwayml/stable-diffusion-v1-5"

ddim_steps=50

echo "Processing with model ID: ${model_id}, seed: ${seed}, view: ${view}, interval: ${interval}"

python ulip_generation.py \
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