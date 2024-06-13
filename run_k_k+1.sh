#!/bin/bash

#SBATCH -p gpu-preempt
#SBATCH -t 04:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --constraint=m40
#SBATCH --mem=50G
#SBATCH --array=0-14
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Activate the conda environment
module load miniconda/22.11.1-1
conda activate control

# Set the prompt
prompt="a photo of a chair"
token=$(echo "$prompt" | tr ' ' '_')
seeds=(549110186 170605 021904) # 091222 270606)
seed_index=$((SLURM_ARRAY_TASK_ID % ${#seeds[@]}))
seed=${seeds[seed_index]}


interval=$((((SLURM_ARRAY_TASK_ID) % 5) + 21))


# Get the interval value based on the array index
echo "Interval: $interval"
# Execute the Python script with the specific prompt and interval
python no_gradio_depth2image_periodic.py \
    --input_image assets/chair_white_diag_cut.png \
    --prompt "$prompt" \
    --a_prompt "" \
    --n_prompt "" \
    --num_samples 1 \
    --image_resolution 512 \
    --detect_resolution 384 \
    --ddim_steps 50 \
    --strength 1.0 \
    --scale 9.0 \
    --seed $seed \
    --eta 0.0 \
    --interval $interval