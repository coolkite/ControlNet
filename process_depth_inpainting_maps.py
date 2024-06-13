import argparse
import os
import random
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline
from PIL import Image
from itertools import product
import gc
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

def get_mask(map):
    _, mask = cv2.threshold(map, 1, 255, cv2.THRESH_BINARY)

    return mask.astype('uint8')

#ignore for now; not being used
def get_foreground(model_id, prompts, chosen_comb, depth_map, output_data_dir, num_samples, image_resolution, ddim_steps, strength, scale):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    prompt_ind, cur_seed, view_ind = chosen_comb
    cur_prompt = prompts[prompt_ind].format("chair")

    with torch.no_grad():
        detected_map = HWC3(depth_map)
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
                                            shape, cond, verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]

        # output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}")
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, f"{category_id}_{model_id_only}_prompt_{prompt_ind}_view_{view_ind}_seed_{cur_seed}.jpg")
        # cv2.imwrite(output_path, cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR))

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    return results[0]
    


def infer(model_id, depth_map_path, foreground_prompts, background_prompts, seeds, views, chosen_combs, num_foreground_steps, num_background_steps, output_data_dir, image_resolution, control_strength=1.2, control_scale_guidance=9, inpaint_strength=1):
    # Load the models
    # controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    background_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    background_pipe.to('cuda')

    # foreground_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    # )
    # foreground_pipe.enable_xformers_memory_efficient_attention()
    # foreground_pipe.to('cuda')

    # Load the depth map
    data = np.load(depth_map_path)["arr_0"]
    
    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    for fore_prompt_ind, back_prompt_ind, cur_seed, view_ind in chosen_combs:
        depth_map = data[view_ind]
        H, W = (image_resolution, image_resolution)

        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Process the depth map and mask
        mask_foreground_numpy = get_mask(depth_map)
        mask_background_numpy = cv2.bitwise_not(mask_foreground_numpy).astype('uint8')
        mask_background = Image.fromarray(mask_background_numpy, 'L')

        foreground_prompt = foreground_prompts[fore_prompt_ind]
        background_prompt = background_prompts[back_prompt_ind]

        # Generate the image
        generator = torch.manual_seed(cur_seed) 
        print(f"Generating image for model {model_id} with foreground prompt {foreground_prompt}, background prompt {background_prompt}, seed {cur_seed}, view {view_ind}...")

        foreground_image_numpy = get_foreground(model_id, foreground_prompts, (fore_prompt_ind, cur_seed, view_ind), depth_map, output_data_dir, 1, image_resolution, num_foreground_steps, control_strength, control_scale_guidance)        
        # masked_foreground_image_numpy = foreground_image_numpy * np.stack([mask_foreground_numpy/255] * 3, axis=-1)
        # convert black pixels to white
        # masked_foreground_image_numpy[mask_foreground_numpy == 0] = 255
        # print(mask_foreground_numpy)
        # print(mask_foreground_numpy.max())
        foreground_image = Image.fromarray(foreground_image_numpy.astype('uint8'), 'RGB')

        # foreground_pipe(
        #     foreground_prompt,
        #     num_inference_steps=num_foreground_steps,
        #     generator=generator,
        #     image=Image.fromarray(depth_map)
        # ).images[0]

        print(num_background_steps)

        background_image = background_pipe(
            background_prompt,
            num_inference_steps=num_background_steps,
            generator=generator,
            image=foreground_image,
            mask_image=mask_background,
            strength=inpaint_strength,
            negative_prompt="poor details, blurry, unstable"
        ).images[0]

        # Save the output image
        combined_output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", "combined", f"{category_id}_{model_id_only}")
        fore_output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", "foreground", f"{category_id}_{model_id_only}")
        os.makedirs(combined_output_dir, exist_ok=True)
        os.makedirs(fore_output_dir, exist_ok=True)
        combined_output_path = os.path.join(combined_output_dir, f"{category_id}_{model_id_only}_foreground_prompt_{fore_prompt_ind}_background_prompt_{back_prompt_ind}_seed_{cur_seed}_view_{view_ind}.jpg")
        fore_output_path = os.path.join(fore_output_dir, f"fore_{category_id}_{model_id_only}_foreground_prompt_{fore_prompt_ind}_background_prompt_{back_prompt_ind}_seed_{cur_seed}_view_{view_ind}.jpg")

        foreground_image.save(fore_output_path)
        background_image.save(combined_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, help="Model ID in the format <category_id>_<model_id>")
    parser.add_argument("--depth_map_path", type=str, required=True, help="Path to the depth map")
    parser.add_argument("--foreground_prompts", nargs="+", required=True, help="List of foreground prompts")
    parser.add_argument("--background_prompts", nargs="+", required=True, help="List of background prompts")
    parser.add_argument("--seed_range", type=int, default=10000, required=True, help="Range of seeds")
    parser.add_argument("--views", nargs="+", type=int, required=True, help="List of views")
    parser.add_argument("--num_samples_per_model", type=int, default=100, help="Number of samples to generate per model")
    parser.add_argument("--num_foreground_steps", type=int, default=20, help="Number of foreground inference steps")
    parser.add_argument("--num_background_steps", type=int, default=50, help="Number of background inference steps")
    parser.add_argument("--output_data_dir", type=str, required=True, help="Path to the output data directory")
    parser.add_argument("--image_resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--control_strength", type=float, default=1.2, help="Control strength")
    parser.add_argument("--control_scale_guidance", type=float, default=9, help="Guidance scale")
    parser.add_argument("--inpaint_strength", type=float, default=1, help="Inpainting strength")

    args = parser.parse_args()

    # Save prompts to a file in the output data directory
    os.makedirs(args.output_data_dir, exist_ok=True)
    try:
        with open(os.path.join(args.output_data_dir, "foreground_prompts.txt"), "w") as f:
            f.write("\n".join(args.foreground_prompts))
        with open(os.path.join(args.output_data_dir, "background_prompts.txt"), "w") as f:
            f.write("\n".join(args.background_prompts))
    except:
        pass

    # Generate combinations
    fore_prompt_inds = list(range(len(args.foreground_prompts)))
    back_prompt_inds = list(range(len(args.background_prompts)))
    seeds = list(range(args.seed_range))

    shape_seed = abs(hash(args.model_id))+1
    rng = np.random.default_rng(shape_seed)

    chosen_combs = []
    for _ in range(args.num_samples_per_model):
        fore_prompt_ind = rng.choice(fore_prompt_inds)
        back_prompt_ind = rng.choice(back_prompt_inds)
        seed = rng.choice(seeds)
        view = rng.choice(args.views)
        chosen_combs.append((fore_prompt_ind, back_prompt_ind, seed, view))

    infer(args.model_id, args.depth_map_path, args.foreground_prompts, args.background_prompts, args.seed_range, args.views, chosen_combs,
          args.num_foreground_steps, args.num_background_steps, args.output_data_dir, args.image_resolution, args.control_strength, args.control_scale_guidance, args.inpaint_strength)