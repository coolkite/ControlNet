import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath("/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/ulip_testing/"))
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
from cldm.ddim_hacked_periodic_no_ctrl import DDIMSampler
import gc

#torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=False)  # Triggers fresh download of MiDaS repo

def extract_depth_midas(input_image, detect_resolution, image_resolution):
    apply_midas = MidasDetector()
    input_image = HWC3(input_image)
    detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    return detected_map

def extract_depth_zoe(input_image, detect_resolution, image_resolution):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    repo = "isl-org/ZoeDepth"
    model_zoe_n = torch.hub.load(repo, "ZoeD_N", pretrained=True)
    apply_zoe = model_zoe_n.to(DEVICE)
    input_image = HWC3(input_image)
    detected_map = apply_zoe.infer_pil(resize_image(input_image, detect_resolution))
    detected_map = HWC3(detected_map)
    img = resize_image(input_image, image_resolution)
    H, W, C = img.shape
    detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
    detected_map = cv2.normalize(detected_map, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    detected_map = detected_map.astype(np.uint8)
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

def load_depth_image(file_path):
    depth_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if depth_image is None:
        raise ValueError(f"Failed to load depth image from {file_path}")
    return depth_image

def generate_images(model_id, input_data_dir, init_prompts, base_prompts, learned_token, learned_token_path, obj, seed, view, output_data_dir, num_samples, image_resolution, ddim_steps, strength, scale, eta, interval):
    model = create_model('../../models/cldm_v15_evaluation_1_5_no_ctrl.yaml').cpu()
    model.load_state_dict(load_state_dict('../../models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    print("Model loaded")

    ddim_sampler = DDIMSampler(model)

    init_prompt = random.choice(init_prompts)
    base_prompt = random.choice(base_prompts)

    with torch.no_grad():
        extra_path = f"{interval}/{view}/{seed}"
        depth_file = os.path.join(input_data_dir, model_id, model_id, f"{model_id}_r_{view:03d}_depth0001.png")
        detected_map = load_depth_image(depth_file)
        detected_map = HWC3(detected_map)
        H, W = (image_resolution, image_resolution)

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)

        print(f"Processing {model_id} with init_prompt {init_prompt}, base_prompt {base_prompt} and view {view} with seed {seed}")

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([init_prompt] * num_samples)], "new_prompt": [base_prompt] * num_samples}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)], "new_prompt": [""] * num_samples}
        
        results_baseline, _, baseline_intermediates = run_diffusion(ddim_sampler, model, cond, un_cond, num_samples, H, W, strength, scale, eta, ddim_steps, interval)
        baseline_x_inter = model.decode_first_stage(baseline_intermediates['x_inter'][interval])

        category_id = model_id.split('-')[0]
        model_id_only = model_id.split('-', 1)[1]
        output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}", extra_path)
        os.makedirs(output_dir, exist_ok=True)
        output_path_baseline = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_baseline.jpg")
        cv2.imwrite(output_path_baseline, cv2.cvtColor(results_baseline[0], cv2.COLOR_RGB2BGR))
        
        baseline_x_inter_img = (einops.rearrange(baseline_x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        output_path_baseline_inter = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_baseline_inter.jpg")
        cv2.imwrite(output_path_baseline_inter, cv2.cvtColor(baseline_x_inter_img[0], cv2.COLOR_RGB2BGR))

        output_path_baseline_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_baseline_depth.jpg")
        generated_depth_img_baseline = extract_depth_zoe(results_baseline[0], 512, 512)
        cv2.imwrite(output_path_baseline_depth, generated_depth_img_baseline)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        seed_everything(seed)
        new_prompt = base_prompt.replace(obj, learned_token)
        print("prompt+idx", new_prompt, base_prompt)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([init_prompt] * num_samples)], "new_prompt": [new_prompt] * num_samples, "learned_embedding":[learned_token, learned_token_path]}
        un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)], "new_prompt": [""] * num_samples}
        
        results_learned, _, learned_intermediates = run_diffusion(ddim_sampler, model, cond, un_cond, num_samples, H, W, strength, scale, eta, ddim_steps, interval)
    
        output_path_learned = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_learned.jpg")
        cv2.imwrite(output_path_learned, cv2.cvtColor(results_learned[0], cv2.COLOR_RGB2BGR))
        learned_x_inter = model.decode_first_stage(learned_intermediates['x_inter'][interval])
        learned_x_inter_img = (einops.rearrange(learned_x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        output_path_learned_inter = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_learned_inter.jpg") 
        cv2.imwrite(output_path_learned_inter, cv2.cvtColor(learned_x_inter_img[0], cv2.COLOR_RGB2BGR))
    
        output_path_learned_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_learned_depth.jpg")
        generated_depth_img_learned = extract_depth_zoe(results_learned[0], 512, 512)
        cv2.imwrite(output_path_learned_depth, generated_depth_img_learned)

        output_path_original_depth = os.path.join(output_dir, f"{category_id}_{model_id_only}_view_{view}_seed_{seed}_interval_{interval}_original_depth.jpg")
        cv2.imwrite(output_path_original_depth, detected_map)

    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID in the format <category_id>-<model_id>")
    parser.add_argument("--init_prompts", nargs="+", required=True, help="List of initial prompts")
    parser.add_argument("--baseline_prompts", nargs="+", default=["a photo of a chair"], help="List of baseline prompts")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--view", type=int, required=True, help="View index")
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
    parser.add_argument("--learned_token", type=str, default="", help="Learned token")
    parser.add_argument("--learned_token_path", type=str, default="", help="Path to the learned token")

    args = parser.parse_args()
    print("model id: ", args.model_id)

    learned_model_id = args.model_id.replace("-", "_")
    learned_token_path = os.path.join(args.learned_token_path, learned_model_id)
    
    print(f"Learned token path: {learned_token_path}")
    print(f"Processing model: {args.model_id}, view: {args.view}, seed: {args.seed}, interval: {args.interval}")
    
    generate_images(args.model_id, args.input_data_dir, args.init_prompts, args.baseline_prompts, args.learned_token, learned_token_path, args.object, 
                    args.seed, args.view, args.output_data_dir, args.num_samples, args.image_resolution,
                    args.ddim_steps, args.strength, args.scale, args.eta, interval=args.interval)

if __name__ == "__main__":
    main()