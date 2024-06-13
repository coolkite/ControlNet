# debug.py

import os

seeds = [100] #, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
torch_seed = 100
prompt = "a photo of a <token1> <token2> <token3> chair" #"a 3d render of a TOKEN"
model = 'stabilityai/stable-diffusion-2-1'
num_diffusion_steps = 50
output_dir = "vis_embeds"

token_path = "/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/learned_embeds/learned_emb_epoch_3000.npy"

for seed in seeds:
    cmd = f"""python eval_injection.py \
              --seed {seed} \
              --prompt '{prompt}' \
              --num_diffusion_steps {num_diffusion_steps} \
              --output_dir {output_dir} \
              --token_path {token_path} \
              --model {model} \
              --torch_seed {torch_seed}"""
    os.system(cmd)