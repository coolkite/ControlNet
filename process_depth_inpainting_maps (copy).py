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

    return Image.fromarray(mask.astype('uint8'), 'L')

#ignore for now; not being used
def get_foreground(model_id, prompts, seeds, views, chosen_comb, input_data_dir, output_data_dir, num_samples, image_resolution, ddim_steps, strength, scale, eta):
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    data = np.load(os.path.join(input_data_dir, model_id))["arr_0"]
    num_views = data.shape[0]

    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    prompt_ind, cur_seed, view_ind = chosen_comb
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

        # output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}")
        # os.makedirs(output_dir, exist_ok=True)
        # output_path = os.path.join(output_dir, f"{category_id}_{model_id_only}_prompt_{prompt_ind}_view_{view_ind}_seed_{cur_seed}.jpg")
        # cv2.imwrite(output_path, cv2.cvtColor(results[0], cv2.COLOR_RGB2BGR))

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
    


def infer(model_id, depth_map_path, foreground_prompts, background_prompts, seeds, views, chosen_combs, num_foreground_steps, num_background_steps, output_data_dir, image_resolution):
    # Load the models
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16)
    background_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        revision="fp16",
        torch_dtype=torch.float16,
        safety_checker=None
    )
    background_pipe.enable_xformers_memory_efficient_attention()
    background_pipe.to('cuda')

    foreground_pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
    )
    foreground_pipe.enable_xformers_memory_efficient_attention()
    foreground_pipe.to('cuda')

    # Load the depth map
    data = np.load(depth_map_path)["arr_0"]
    
    category_id = model_id.split('_')[0]
    model_id_only = model_id.split('_')[1]

    for fore_prompt_ind, back_prompt_ind, cur_seed, view_ind in chosen_combs:
        depth_map = data[view_ind]
        H, W = (image_resolution, image_resolution)

        depth_map = cv2.resize(depth_map, (W, H), interpolation=cv2.INTER_LINEAR)

        # Process the depth map and mask
        mask_foreground = get_mask(depth_map)
        mask_background = Image.fromarray(cv2.bitwise_not(np.array(mask_foreground)).astype('uint8'), 'L')

        foreground_prompt = foreground_prompts[fore_prompt_ind].format("chair")
        background_prompt = background_prompts[back_prompt_ind]

        # Generate the image
        generator = torch.manual_seed(cur_seed) 
        print(f"Generating image for model {model_id} with foreground prompt {foreground_prompt}, background prompt {background_prompt}, seed {cur_seed}, view {view_ind}...")

        foreground_image = foreground_pipe(
            foreground_prompt,
            num_inference_steps=num_foreground_steps,
            generator=generator,
            image=Image.fromarray(depth_map)
        ).images[0]

        background_image = background_pipe(
            background_prompt,
            num_inference_steps=num_background_steps,
            generator=generator,
            image=foreground_image,
            mask_image=mask_background
        ).images[0]

        # Save the output image
        output_dir = os.path.join(output_data_dir, f"{category_id}_pointbert_100_controlnet", f"{category_id}_{model_id_only}")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{category_id}_{model_id_only}_foreground_prompt_{fore_prompt_ind}_background_prompt_{back_prompt_ind}_seed_{cur_seed}_view_{view_ind}.jpg")
        fore_output_path = os.path.join(output_dir, f"fore_{category_id}_{model_id_only}_foreground_prompt_{fore_prompt_ind}_background_prompt_{back_prompt_ind}_seed_{cur_seed}_view_{view_ind}.jpg")

        foreground_image.save(fore_output_path)
        background_image.save(output_path)

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

    shape_seed = abs(hash(args.model_id))
    rng = np.random.default_rng(shape_seed)

    combs = list(product(*(fore_prompt_inds, back_prompt_inds, seeds, args.views)))
    rng.shuffle(combs)
    chosen_combs = combs[:args.num_samples_per_model]

    infer(args.model_id, args.depth_map_path, args.foreground_prompts, args.background_prompts, args.seed_range, args.views, chosen_combs,
          args.num_foreground_steps, args.num_background_steps, args.output_