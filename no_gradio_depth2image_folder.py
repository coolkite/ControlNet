from share import *
import config

import cv2
import einops
import numpy as np
import torch
import os
import random
import argparse

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked_periodic import DDIMSampler

def process(input_image_path, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, output_folder, interval=1):
    apply_midas = MidasDetector()

    model = create_model('./models/cldm_v15_periodic.yaml').cpu()
    model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    with torch.no_grad():
        input_image = cv2.imread(input_image_path)
        input_image = HWC3(input_image)
        detected_map, _ = apply_midas(resize_image(input_image, detect_resolution))
        detected_map = HWC3(detected_map)
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "c_crossattn_chair": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)], "c_crossattn_stool": [model.get_learned_conditioning(["a photo of a stool" + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond,
                                                     interval=interval)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        intermediates['detected_maps'] = [detected_map] * (ddim_steps - 25)

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the generated images in the output folder
        for i, image in enumerate(results):
            output_path = os.path.join(output_folder, f"{prompt}_{seed}_depth_{interval}.png")
            cv2.imwrite(output_path, image)

        create_output_video(model, intermediates, results, ddim_steps, detected_map, f"{prompt}_{seed}", output_folder, interval)

    return results

def create_output_video(model, intermediates, results, ddim_steps, mask, name, output_folder, interval=1):
    final_image = results[0]

    video_path = os.path.join(output_folder, f"{name}_depth_{interval}.mp4")
    frame_size = (final_image.shape[1] * 4, final_image.shape[0])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, frame_size)

    for i in range(len(intermediates['x_inter'])):
        x_inter = model.decode_first_stage(intermediates['x_inter'][i])
        pred_x0 = model.decode_first_stage(intermediates['pred_x0'][i])

        x_inter = (einops.rearrange(x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        pred_x0 = (einops.rearrange(pred_x0, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        resized_mask = cv2.resize(mask, (x_inter.shape[1], x_inter.shape[0]))

        frame = np.zeros((x_inter.shape[0], x_inter.shape[1] * 4, 3), dtype=np.uint8)

        if len(resized_mask.shape) == 2:
            frame[:, :x_inter.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        else:
            frame[:, :x_inter.shape[1], :] = resized_mask

        frame[:, x_inter.shape[1]:x_inter.shape[1]*2, :] = cv2.cvtColor(x_inter, cv2.COLOR_BGR2RGB)
        frame[:, x_inter.shape[1]*2:x_inter.shape[1]*3, :] = cv2.cvtColor(pred_x0, cv2.COLOR_BGR2RGB)

        frame[:, x_inter.shape[1]*3:, :] = np.zeros((x_inter.shape[0], x_inter.shape[1], 3), dtype=np.uint8)

        cv2.putText(frame, f"DDIM Step: {i+1}/{ddim_steps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        video_writer.write(frame)

    resized_mask = cv2.resize(mask, (final_image.shape[1], final_image.shape[0]))
    final_frame = np.zeros((final_image.shape[0], final_image.shape[1] * 4, 3), dtype=np.uint8)

    if len(resized_mask.shape) == 2:
        final_frame[:, :final_image.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    else:
        final_frame[:, :final_image.shape[1], :] = resized_mask

    final_frame[:, final_image.shape[1]:final_image.shape[1]*2, :] = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_frame[:, final_image.shape[1]*2:final_image.shape[1]*3, :] = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    final_frame[:, final_image.shape[1]*3:, :] = np.zeros((final_image.shape[0], final_image.shape[1], 3), dtype=np.uint8)

    for _ in range(10):
        video_writer.write(final_frame)

    video_writer.release()
    print(f"Video saved in {video_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the image generation")
    parser.add_argument("--a_prompt", type=str, default="best quality, extremely detailed", help="Additional prompt")
    parser.add_argument("--n_prompt", type=str, default="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality", help="Negative prompt")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--image_resolution", type=int, default=512, help="Image resolution")
    parser.add_argument("--detect_resolution", type=int, default=384, help="Depth detection resolution")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Number of DDIM steps")
    parser.add_argument("--guess_mode", action="store_true", help="Enable guess mode")
    parser.add_argument("--strength", type=float, default=1.0, help="Control strength")
    parser.add_argument("--scale", type=float, default=9.0, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=549110186, help="Random seed")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta")
    parser.add_argument("--interval", type=int, default=1, help="Interval for generating video frames")
    parser.add_argument("--output_folder", type=str, default="results", help="Output folder for generated images")
    args = parser.parse_args()

    process(args.input_image, args.prompt, args.a_prompt, args.n_prompt, args.num_samples, args.image_resolution,
            args.detect_resolution, args.ddim_steps, args.guess_mode, args.strength, args.scale, args.seed, args.eta,
            args.output_folder, args.interval)

if __name__ == "__main__":
    main()