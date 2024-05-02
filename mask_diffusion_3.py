from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os, datetime, base64

from PIL import Image
from io import BytesIO

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked_mask_gradual import DDIMSampler_Mask_Gradual
from cldm.ddim_hacked import DDIMSampler

apply_canny = CannyDetector()

# model_normal = create_model('./models/cldm_v15.yaml').cpu()
model_masked = create_model('./models/cldm_v15_mask.yaml').cpu()
print("loading state dict")
# model_normal.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model_masked.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
# model_normal = model_normal.cuda()
model_masked = model_masked.cuda()

#ddim_sampler_normal = DDIMSampler(model_normal)
ddim_sampler_masked = DDIMSampler_Mask_Gradual(model_masked)


def create_output_video(model, intermediates, results, ddim_steps, mask, name):
    # Save the final image
    final_image = results[0]

    # Create a video from the intermediate images
    log_dir = "saved_vids_mask"
    os.makedirs(log_dir, exist_ok=True)
    print(f"{name}.mp4")
    video_path = os.path.join(log_dir, f"{name}.mp4")
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
        frame[:, :x_inter.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
        frame[:, x_inter.shape[1]:x_inter.shape[1]*2, :] = x_inter
        frame[:, x_inter.shape[1]*2:x_inter.shape[1]*3, :] = pred_x0

        print(x_inter.shape)
        if i > 25:
            print(len(intermediates['detected_maps']))
            print(i)
            detected_map = intermediates['detected_maps'][i - 25 - 2]
            frame[:, x_inter.shape[1]*3:, :] = cv2.cvtColor(detected_map, cv2.COLOR_GRAY2BGR)
        else:
            frame[:, x_inter.shape[1]*3:, :] = np.zeros((512, 512, 3), dtype=np.uint8)

        # Add ddim_step value at the top of the frame
        cv2.putText(frame, f"DDIM Step: {i+1}/{ddim_steps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame to the video
        video_writer.write(frame)

    # Add the final image to the video
    resized_mask = cv2.resize(mask, (final_image.shape[1], final_image.shape[0]))
    final_frame = np.zeros((final_image.shape[0], final_image.shape[1] * 4, 3), dtype=np.uint8)
    final_frame[:, :final_image.shape[1], :] = cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR)
    final_frame[:, final_image.shape[1]:final_image.shape[1]*2, :] = final_image
    final_frame[:, final_image.shape[1]*2:final_image.shape[1]*3, :] = final_image
    if len(intermediates['detected_maps']) > 0:
        final_frame[:, final_image.shape[1]*3:, :] = cv2.cvtColor(intermediates['detected_maps'][-1], cv2.COLOR_GRAY2BGR)
    else:
        final_frame[:, final_image.shape[1]*3:, :] = np.zeros((final_image.shape[0], final_image.shape[1], 3), dtype=np.uint8)


    for _ in range(10):  # Show the final image for 1 second (10 frames)
        video_writer.write(final_frame)

    video_writer.release()
    print(f"Video saved in {video_path}")

def log_run(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, output_images):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "run_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"2_masked_diffusion_gradual.html")

    with open(log_file, "a") as f:
        f.write("<html><head><title>Run Log</title></head><body>\n")
        f.write(f"<h2>Run Timestamp: {timestamp}</h2>\n")
        f.write("<h3>Inputs:</h3>\n")
        f.write(f"<p>Prompt: {prompt}</p>\n")
        f.write(f"<p>Additional Prompt: {a_prompt}</p>\n")
        f.write(f"<p>Negative Prompt: {n_prompt}</p>\n")
        f.write(f"<p>Number of Samples: {num_samples}</p>\n")
        f.write(f"<p>Image Resolution: {image_resolution}</p>\n")
        f.write(f"<p>DDIM Steps: {ddim_steps}</p>\n")
        f.write(f"<p>Guess Mode: {guess_mode}</p>\n")
        f.write(f"<p>Strength: {strength}</p>\n")
        f.write(f"<p>Scale: {scale}</p>\n")
        f.write(f"<p>Seed: {seed}</p>\n")
        f.write(f"<p>Eta: {eta}</p>\n")
        f.write(f"<p>Low Threshold: {low_threshold}</p>\n")
        f.write(f"<p>High Threshold: {high_threshold}</p>\n")

        f.write("<h3>Input Image:</h3>\n")
        input_image_resized = Image.fromarray(input_image).resize((256, 256))
        f.write(f'<img src="data:image/png;base64,{image_to_base64(input_image_resized)}" alt="Input Image"><br>\n')

        f.write("<h3>Output Images:</h3>\n")
        f.write('<table><tr>\n')
        for i, output_image in enumerate(output_images):
            output_image_resized = Image.fromarray(output_image).resize((256, 256))
            f.write(f'<td><img src="data:image/png;base64,{image_to_base64(output_image_resized)}" alt="Output Image {i+1}"></td>\n')
        f.write("</tr></table><br>\n")
        f.write("</body></html>\n")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def scale_image_and_masks(image, mask, orig_res, scale_factor=1, translation_factor=(0,0), rotation_angle=0):
    original_size = image.shape[:2]
    new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
    scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(mask, new_size, interpolation=cv2.INTER_AREA)

    # Create a new blank image with the original size and paste the scaled image at the center
    scaled_image_final = np.zeros((original_size[0], original_size[1], image.shape[2]), dtype=np.uint8)
    scaled_mask_final = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)

    # Calculate the translation offset
    translation_offset = (int(original_size[1] * translation_factor[0]), int(original_size[0] * translation_factor[1]))

    # Apply translation
    paste_position = (
        max(0, (original_size[1] - new_size[1]) // 2 + translation_offset[0]),
        max(0, (original_size[0] - new_size[0]) // 2 + translation_offset[1])
    )
    paste_end_position = (
        min(paste_position[0] + new_size[0], original_size[1]),
        min(paste_position[1] + new_size[1], original_size[0])
    )
    scaled_image_final[paste_position[1]:paste_end_position[1], paste_position[0]:paste_end_position[0]] = \
        scaled_image[:paste_end_position[1]-paste_position[1], :paste_end_position[0]-paste_position[0]]
    scaled_mask_final[paste_position[1]:paste_end_position[1], paste_position[0]:paste_end_position[0]] = \
        scaled_mask[:paste_end_position[1]-paste_position[1], :paste_end_position[0]-paste_position[0]]

    # Apply rotation
    center = (original_size[1] // 2, original_size[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    scaled_image_final = cv2.warpAffine(scaled_image_final, rotation_matrix, (original_size[1], original_size[0]))
    scaled_mask_final = cv2.warpAffine(scaled_mask_final, rotation_matrix, (original_size[1], original_size[0]))

    scaled_image_final = cv2.bitwise_and(scaled_image_final, scaled_image_final, mask=scaled_mask_final)
    scaled_image_final[scaled_mask_final == 0] = 0

    scaled_image_final = cv2.resize(scaled_image_final, (orig_res, orig_res), interpolation=cv2.INTER_AREA)
    scaled_mask_final = cv2.resize(scaled_mask_final, (orig_res, orig_res), interpolation=cv2.INTER_AREA)

    # scaled_image_final = Image.fromarray(scaled_image_final.astype('uint8'), 'RGB')
    # scaled_mask_final = Image.fromarray(scaled_mask_final.astype('uint8'), 'L')

    return scaled_image_final, scaled_mask_final

def get_mask(image):
    image = np.array(image)
    #image = cv2.resize(image, (image.shape[0]//8, image.shape[1]//8), interpolation=cv2.INTER_AREA)
    if image.shape[2] == 4:
            mask = image[:, :, 3]
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    else:
        foreground_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(foreground_gray, 254, 255, cv2.THRESH_BINARY_INV)
        #_, mask = cv2.threshold(foreground_gray, 1, 255, cv2.THRESH_BINARY)

    #return Image.fromarray(mask.astype('uint8'), 'L')
    return mask

def process(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strengths, scale, seed, eta, low_threshold, high_threshold):
    n_result = []
    m_result = []
    for strength in [strengths]: #, strengths*1.5, strengths*2]:
        with torch.no_grad():
            img = resize_image(HWC3(input_image), image_resolution)
            mask = get_mask(input_image)
            img, mask = scale_image_and_masks(img,mask,image_resolution,0.5)
            H, W, C = img.shape

            #detected_map = apply_canny(img, low_threshold, high_threshold)
            detected_map = apply_canny(cv2.bitwise_and(img, img, mask=mask), low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
            print("detected map: ", detected_map.shape)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()
            print("control: ", control.shape)

            mask = cv2.resize(mask, (mask.shape[0]//8, mask.shape[1]//8), interpolation=cv2.INTER_AREA)
            input_mask = torch.from_numpy(mask.copy().reshape(H//8, W//8, 1)).float().cuda() / 255.0
            input_mask = torch.stack([input_mask for _ in range(num_samples)], dim=0)
            input_mask = einops.rearrange(input_mask, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                #model_normal.low_vram_shift(is_diffusing=False)
                model_masked.low_vram_shift(is_diffusing=False)

            # cond_norm = {"c_concat": [control], "c_crossattn": [model_normal.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            # un_cond_norm = {"c_concat": None if guess_mode else [control], "c_crossattn": [model_normal.get_learned_conditioning([n_prompt] * num_samples)]}
            # shape = (4, H // 8, W // 8)

            # model_normal.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            # samples_normal, intermediates = ddim_sampler_normal.sample(ddim_steps, num_samples,
            #                                             shape, cond_norm, verbose=False, eta=eta,
            #                                             unconditional_guidance_scale=scale,
            #                                             unconditional_conditioning=un_cond_norm,
            #                                             )
    
            cond_mask = {"c_concat": [control], "c_crossattn": [model_masked.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond_mask = {"c_concat": None if guess_mode else [control], "c_crossattn": [model_masked.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)
            model_masked.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples_masked, intermediates = ddim_sampler_masked.sample(ddim_steps, num_samples,
                                                        shape, cond_mask, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond_mask,
                                                        mask=input_mask, detector=apply_canny, decoder=model_masked.decode_first_stage, GRADUAL_THRESH=25)

            # x_samples_normal = model_normal.decode_first_stage(samples_normal)
            # x_samples_normal = (einops.rearrange(x_samples_normal, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            # results_normal = [x_samples_normal[i] for i in range(num_samples)]
            print(samples_masked.shape)
            x_samples_masked = model_masked.decode_first_stage(samples_masked)
            x_samples_masked = (einops.rearrange(x_samples_masked, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            results_masked = [x_samples_masked[i] for i in range(num_samples)]

            # n_result = n_result + results_normal
            m_result = m_result + results_masked

        output_images = [detected_map] + m_result + [mask]

        log_run(input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold, output_images)
        create_output_video(model_masked, intermediates, results_masked, ddim_steps, mask, prompt.replace(" ", "_")+str(seed))

    return output_images


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Stable Diffusion with Canny Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")
            prompt = gr.Textbox(label="Prompt", value="A chair in a living room")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                low_threshold = gr.Slider(label="Canny low threshold", minimum=1, maximum=255, value=100, step=1)
                high_threshold = gr.Slider(label="Canny high threshold", minimum=1, maximum=255, value=200, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=10, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=762040392)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt",
                                      value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality')
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', share=True)