from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import os
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.midas import MidasDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_midas = MidasDetector()

model = create_model('./models/cldm_v21.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_depth.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
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

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        # Save intermediates
        intermediates['detected_maps'] = [detected_map] * (ddim_steps - 25)

        # Create video of intermediates
        create_output_video(model, intermediates, results, ddim_steps, detected_map, f"{prompt}_{seed}")

    return [detected_map] + results

def create_output_video(model, intermediates, results, ddim_steps, mask, name):
    # Save the final image
    final_image = results[0]

    # Create a video from the intermediate images
    log_dir = "sample_run_meeting"
    os.makedirs(log_dir, exist_ok=True)
    print(f"{name}.mp4")
    video_path = os.path.join(log_dir, f"{name}_depth.mp4")
    frame_size = (final_image.shape[1] * 4, final_image.shape[0])  # Adjust frame size for three columns
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, 10, frame_size)

    print(len(intermediates['x_inter']))
    for i in range(len(intermediates['x_inter'])):
        # Decode x_inter and pred_x0
        x_inter = model.decode_first_stage(intermediates['x_inter'][i])
        pred_x0 = model.decode_first_stage(intermediates['pred_x0'][i])

        # Convert to numpy arrays and scale
        x_inter = (einops.rearrange(x_inter, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]
        pred_x0 = (einops.rearrange(pred_x0, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)[0]

        # Resize mask to match the image size
        resized_mask = cv2.resize(mask, (x_inter.shape[1], x_inter.shape[0]))

        # Create a frame with three columns
        frame = np.zeros((x_inter.shape[0], x_inter.shape[1] * 4, 3), dtype=np.uint8)

        if len(resized_mask.shape) == 2:
            frame[:, :x_inter.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        else:
            frame[:, :x_inter.shape[1], :] = resized_mask

        frame[:, x_inter.shape[1]:x_inter.shape[1]*2, :] = cv2.cvtColor(x_inter, cv2.COLOR_BGR2RGB)
        frame[:, x_inter.shape[1]*2:x_inter.shape[1]*3, :] = cv2.cvtColor(pred_x0, cv2.COLOR_BGR2RGB)

        frame[:, x_inter.shape[1]*3:, :] = np.zeros((x_inter.shape[0], x_inter.shape[1], 3), dtype=np.uint8)

        # Add ddim_step value at the top of the frame
        cv2.putText(frame, f"DDIM Step: {i+1}/{ddim_steps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the video
        video_writer.write(frame)

    # Add the final image to the video
    resized_mask = cv2.resize(mask, (final_image.shape[1], final_image.shape[0]))
    final_frame = np.zeros((final_image.shape[0], final_image.shape[1] * 4, 3), dtype=np.uint8)

    if len(resized_mask.shape) == 2:
        final_frame[:, :final_image.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    else:
        final_frame[:, :final_image.shape[1], :] = resized_mask

    final_frame[:, final_image.shape[1]:final_image.shape[1]*2, :] = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    final_frame[:, final_image.shape[1]*2:final_image.shape[1]*3, :] = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)

    final_frame[:, final_image.shape[1]*3:, :] = np.zeros((final_image.shape[0], final_image.shape[1], 3), dtype=np.uint8)

    for _ in range(10):  # Show the final image for 1 second (10 frames)
        video_writer.write(final_frame)

    video_writer.release()
    print(f"Video saved in {video_path}")

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Depth Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                detect_resolution = gr.Slider(label="Depth Resolution", minimum=128, maximum=1024, value=384, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=549110186)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', share=True)
