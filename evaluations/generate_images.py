import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/baseline_learned_control_k.py"))
print(SCRIPT_DIR)
sys.path.append(os.path.dirname(SCRIPT_DIR))

import random
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector

from cldm.model import create_model, load_state_dict
from ControlNet.cldm.ddim_hacked_periodic_2_1 import DDIMSampler
from itertools import product
import gc

def extract_depth(input_image, detect_resolution, image_resolution):
    apply_midas = MidasDetector()
    input_image = HWC3(input_image)
    detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape

    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return detected_map

def run_diffusion(ddim_sampler, model, cond, un_cond, num_samples, H, W, strength, scale, eta, ddim_steps, interval):
    shape = (4, H // 8, W // 8)

    model.control_scales = ([strength] * 13)
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                     shape, cond, verbose=False, eta=eta,
                                     unconditional_guidance_scale=scale,
                                     unconditional_conditioning=un_cond,
                                     interval=interval)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    results = [x_samples[i] for i in range(num_samples)]
    return results, samples, intermediates

def generate_images(model_id, input_data, init_prompts, base_prompts, learned_token, learned_token_path, object, chosen_combs, output_data_dir, num_samples, image_resolution, ddim_steps, strength, scale, eta, interval=1):
    model = create_model('../models/cldm_v15_evaluation.yaml').cpu()
    model.load_state_dict(load_state_dict('../models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    print("Model loaded")

    second_model_sd = create_model('../models/sd_2_1_inference.yaml').cpu()
    second_model_sd.load_state_dict(load_state_dict('stable-diffusion-2-1/v2-1_768-ema-pruned.safetensors', location='cuda'))
    second_model_sd = second_model_sd.cuda()
    print("Second model loaded")

    ddim_sampler = DDIMSampler(model, second_model_sd)

    print(f"Input data shape: {input_data.shape}")

    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    for init_prompt_ind, base_prompt_ind, cur_seed, view_ind in chosen_combs:
        init_prompt = init_prompts[init_prompt_ind]
        base_prompt = base_prompts[base_prompt_ind]

        with torch.no_grad():
            extra_path = f"{interval}/{view_ind}/{cur_seed}"
            detected_map = HWC3(input_data[view_ind])
            H, W = (image_resolution, image_resolution)

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed_everything(cur_seed)

            print(f"Processing {category_id}_{model_id_only} with init_prompt {init_prompt}, base_prompt {base_prompt} and view {view_ind} with seed {cur_seed}")

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([init_prompt] * num_samples)], "c_crossattn_no_ctrl": [second_model_sd.get_learned_conditioning([base_prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)], "c_crossattn_no_ctrl": [second_model_sd.get_learned_conditioning([""] * num_samples)]}

            results_baseline, _, baseline_intermediates = run_diffusion(ddim_sampler, model, cond, un_cond, num_samples, H, W, strength, scale, eta, ddim_steps, interval)
            baseline_x_inter = model.decode_first_stage(baseline_intermediates['x_inter'][interval])

            output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}", extra_path)
            os.makedirs(output_dir, exist_ok=True)
            output_path_baseline = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_baseline.jpg")
            cv2.imwrite(output_path_baseline, cv2.cvtColor(results_baseline[0], cv2.COLOR_RGB2BGR))
            
            baseline_x_inter_img = (einops.rearrange(baseline_x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            output_path_baseline_inter = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_baseline_inter.jpg")
            cv2.imwrite(output_path_baseline_inter, cv2.cvtColor(baseline_x_inter_img[0], cv2.COLOR_RGB2BGR))

            #create_output_video(model, baseline_intermediates, results_baseline, ddim_steps, detected_map, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_baseline", output_dir, interval=interval)

            output_path_baseline_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_baseline_depth.jpg")
            generated_depth_img_baseline = extract_depth(results_baseline[0], 512, 512)
            cv2.imwrite(output_path_baseline_depth, generated_depth_img_baseline)

            seed_everything(cur_seed)
            object_idx = base_prompt.find(object)
            new_prompt = base_prompt.replace(object, learned_token)
            print(new_prompt)
            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([init_prompt] * num_samples)], "c_crossattn_no_ctrl": [second_model_sd.get_learned_conditioning([new_prompt] * num_samples, learned_embedding_w_object_idx=[learned_token, learned_token_path, object_idx])]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)], "c_crossattn_no_ctrl": [second_model_sd.get_learned_conditioning([""] * num_samples)]}

            results_learned, _, learned_intermediates = run_diffusion(ddim_sampler, model, cond, un_cond, num_samples, H, W, strength, scale, eta, ddim_steps, interval)
        
            output_path_learned = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_learned.jpg")
            cv2.imwrite(output_path_learned, cv2.cvtColor(results_learned[0], cv2.COLOR_RGB2BGR))
            learned_x_inter = model.decode_first_stage(learned_intermediates['x_inter'][interval])
            learned_x_inter_img = (einops.rearrange(learned_x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            output_path_learned_inter = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_learned_inter.jpg") 
            cv2.imwrite(output_path_learned_inter, cv2.cvtColor(learned_x_inter_img[0], cv2.COLOR_RGB2BGR))
        
            
            #create_output_video(model, learned_intermediates, results_learned, ddim_steps, detected_map, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_learned", output_dir, interval=interval)

            output_path_learned_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_learned_depth.jpg")
            generated_depth_img_learned = extract_depth(results_learned[0], 512, 512)
            cv2.imwrite(output_path_learned_depth, generated_depth_img_learned)

            #Save input depth
            output_path_original_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_init_{init_prompt_ind}_base_{base_prompt_ind}_view_{view_ind}_seed_{cur_seed}_interval_{interval}_original_depth.jpg")
            cv2.imwrite(output_path_original_depth, detected_map)

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

def main():
    print("Starting arg parse")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID in the format <category_id>_<model_id>")
    parser.add_argument("--init_prompts", nargs="+", required=True, help="List of prompts")
    parser.add_argument("--seed_range", type=int, default=10000, required=True, help="List of seeds")
    parser.add_argument("--views", nargs="+", type=int, required=True, help="List of views")
    parser.add_argument("--num_samples_per_model", type=int, default=20, help="Number of samples to generate per model")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Path to the input data directory")
    parser.add_argument("--output_data_dir", type=str, required=True, help="Path to the output data directory")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--image_resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength")
    parser.add_argument("--scale", type=float, default=9.0, help="Guidance scale")
    parser.add_argument("--eta", type=float, default=0.0, help="eta (DDIM)")
    parser.add_argument("--object", type=str, default="chair", help="Object")
    parser.add_argument("--interval", type=int, default=1, help="Interval")
    parser.add_argument("--baseline_prompts", nargs="+", default=["a photo of a chair"], help="Base prompts")
    parser.add_argument("--learned_token", type=str, default="", help="learned token")
    parser.add_argument("--learned_token_path", type=str, default="", help="Path to the learned token")

    args = parser.parse_args()
    print("args parsed")

    # Save prompts to a file in the output data directory
    os.makedirs(args.output_data_dir, exist_ok=True)
    try:
        with open(os.path.join(args.output_data_dir, "init_prompts.txt"), "w") as f:
            f.write("\n".join(args.init_prompts))
        print("Initial prompts saved to prompts.txt")
    except:
        pass

    try:
        with open(os.path.join(args.output_data_dir, "base_prompts.txt"), "w") as f:
            f.write("\n".join(args.baseline_prompts))
        print("Base prompts saved to base_prompts.txt")
    except:
        pass

    init_prompt_inds = list(range(len(args.init_prompts)))
    base_prompt_inds = list(range(len(args.baseline_prompts)))
    seeds = list(range(args.seed_range))
    views = args.views

    shape_seed = abs(hash(args.model_id))
    rng = np.random.default_rng(shape_seed)

    chosen_combs = []
    model_ids = ["03001627_1bec15f362b641ca7350b1b2f753f3a2", 
                 "03001627_1157d8d6995da5c0290d57214c8512a4", 
                 "03001627_1f5a2c231265aa9380b3cfbeccfb24d2",
                 "03001627_2a417b5a946ff7eb2a3f8f484e6c5c4f",
                 "03001627_8e2b44ec14701d057c2b071b8bda1b69"
                 ]
    seeds = [170605, 21904]
    views = [0, 7, 15]
    for _ in range(args.num_samples_per_model):
        for seed in seeds:
            for view in views:
                init_prompt_idx = rng.choice(init_prompt_inds)
                base_prompt_inx = rng.choice(base_prompt_inds)
                # seed = rng.choice(seeds)
                # view = rng.choice(args.views)
                chosen_combs.append((init_prompt_idx, base_prompt_inx, seed, view))

    input_data = np.load(os.path.join(args.input_data_dir, args.model_id+"_depth_views.npz"))["arr_0"]
    #learned_token_path_shape_name = args.model_id.replace("_depth_views.npz", "")
    learned_token_path = os.path.join(args.learned_token_path, args.model_id) #f"/project/pi_ekalogerakis_umass_edu/dmpetrov/data/textual_inversion_results/{learned_token_path_shape_name}/learned_embeds.safetensors"
    
    print(learned_token_path)
    print(chosen_combs)


    "/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/data/03001627_pointbert_100_depth"
    
    # Generate images for evaluation
    generate_images(args.model_id, input_data, args.init_prompts, args.baseline_prompts, args.learned_token, learned_token_path, args.object, chosen_combs, 
                    args.output_data_dir, args.num_samples, args.image_resolution,
                    args.ddim_steps, args.strength, args.scale, args.eta, interval=args.interval)

if __name__ == "__main__":
    main()