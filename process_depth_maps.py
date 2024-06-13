import argparse
import os
import random
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from itertools import product
import gc


def process(model_id, prompts, seeds, views, chosen_combs, input_data_dir, output_data_dir, num_samples, image_resolution, ddim_steps, strength, scale, eta):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    data = np.load(os.path.join(input_data_dir, model_id))["arr_0"]
    num_views = data.shape[0]

    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    for prompt_ind, cur_seed, view_ind in chosen_combs:
        cur_prompt = prompts[prompt_ind].format("chair")

        with torch.no_grad():
            detected_map = HWC3(data[view_ind])
            H, W = (image_resolution, image_resolution)

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            seed_everything(cur_seed)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([cur_prompt] * num_samples)]}
            un_cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([""] * num_samples)]}
            shape = (4, H // 8, W // 8)

            model.control_scales = ([strength] * 13)
            samples, _ = ddim_sampler.sample(ddim_steps, num_samples,
                                             shape, cond, verbose=False, eta=eta,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=un_cond)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

            results = [x_samples[i] for i in range(num_samples)]

            output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{category_id}_{model_id_only}_prompt_{prompt_ind}_view_{view_ind}_seed_{cur_seed}.jpg")
            cv2.imwrite(output_path, cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR))

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID in the format <category_id>_<model_id>")
    parser.add_argument("--prompts", nargs="+", required=True, help="List of prompts")
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

    args = parser.parse_args()

    # Save prompts to a file in the output data directory
    os.makedirs(args.output_data_dir, exist_ok=True)
    try:
        with open(os.path.join(args.output_data_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(args.prompts))
        print("Prompts saved to prompts.txt")
    except:
        pass

    prompt_inds = list(range(len(args.prompts)))
    seeds = list(range(args.seed_range))
    views = args.views

    shape_seed = abs(hash(args.model_id))
    rng = np.random.default_rng(shape_seed)

    combs = list(product(*(prompt_inds, seeds, views)))
    rng.shuffle(combs)
    chosen_combs = combs[:args.num_samples_per_model]

    process(args.model_id, args.prompts, args.seed_range, args.views, chosen_combs,
            args.input_data_dir, args.output_data_dir, args.num_samples, args.image_resolution,
            args.ddim_steps, args.strength, args.scale, args.eta)

if __name__ == "__main__":
    main()