from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline
from PIL import Image
import torch
import gradio as gr
import numpy as np
import cv2
import datetime
import os
from io import BytesIO
import base64



# Load the models (assuming they are already downloaded and available locally)
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
# background_pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
#      "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None
#  )
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

def get_canny_image(image, low_threshold, high_threshold):
    canny_image = cv2.Canny(np.array(image), low_threshold, high_threshold)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    return canny_image

def get_mask(image):
    image = np.array(image)
    if image.shape[2] == 4:
            mask = image[:, :, 3]
            mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    else:
        foreground_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(foreground_gray, 1, 255, cv2.THRESH_BINARY)

    return Image.fromarray(mask.astype('uint8'), 'L')

def log_run(image, foreground_prompt, background_prompt, num_foreground_steps, num_background_steps, foreground_scale, translation_x, translation_y, rotation_angle, seed, output_images):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = "run_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "inpaint_background.html")

    with open(log_file, "a") as f:
        f.write("<html><head><title>Run Log</title></head><body>\n")
        f.write(f"<h2>Run Timestamp: {timestamp}</h2>\n")
        f.write("<h3>Inputs:</h3>\n")
        f.write(f"<p>Foreground Prompt: {foreground_prompt}</p>\n")
        f.write(f"<p>Background Prompt: {background_prompt}</p>\n")
        f.write(f"<p>Num Foreground Steps: {num_foreground_steps}</p>\n")
        f.write(f"<p>Num Background Steps: {num_background_steps}</p>\n")
        f.write(f"<p>Foreground Scale: {foreground_scale}</p>\n")
        f.write(f"<p>Translation X: {translation_x}</p>\n")
        f.write(f"<p>Translation Y: {translation_y}</p>\n")
        f.write(f"<p>Rotation Angle: {rotation_angle}</p>\n")
        f.write(f"<p>Seed: {seed}</p>\n")
        f.write("<h3>Input Image:</h3>\n")
        f.write(f'<img src="data:image/png;base64,{image_to_base64(Image.fromarray(image).resize((256, 256)))}" alt="Input Image"><br>\n')
        f.write("<h3>Output Images:</h3>\n")
        f.write('<table><tr style="overflow-x: auto; white-space: nowrap;">\n')
        for i, output_image in enumerate(output_images):
            f.write(f'<td><img src="data:image/png;base64,{image_to_base64(output_image.resize((256, 256)))}" alt="Output Image {i+1}"></td>\n')
        f.write("</tr></table><br>\n")
        f.write("</body></html>\n")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def scale_image_and_masks(image, mask_foreground, mask_background, scale_factor, translation_factor, rotation_angle):
    original_size = image.shape[:2]
    new_size = (int(original_size[1] * scale_factor), int(original_size[0] * scale_factor))
    scaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    scaled_mask_foreground = cv2.resize(mask_foreground, new_size, interpolation=cv2.INTER_AREA)
    scaled_mask_background = cv2.resize(mask_background, new_size, interpolation=cv2.INTER_AREA)

    # Create a new blank image with the original size and paste the scaled image at the center
    scaled_image_final = np.zeros((original_size[0], original_size[1], image.shape[2]), dtype=np.uint8)
    scaled_mask_foreground_final = np.zeros((original_size[0], original_size[1]), dtype=np.uint8)

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
    scaled_mask_foreground_final[paste_position[1]:paste_end_position[1], paste_position[0]:paste_end_position[0]] = \
        scaled_mask_foreground[:paste_end_position[1]-paste_position[1], :paste_end_position[0]-paste_position[0]]

    # Apply rotation
    center = (original_size[1] // 2, original_size[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    scaled_image_final = cv2.warpAffine(scaled_image_final, rotation_matrix, (original_size[1], original_size[0]))
    scaled_mask_foreground_final = cv2.warpAffine(scaled_mask_foreground_final, rotation_matrix, (original_size[1], original_size[0]))

    scaled_image_final = cv2.resize(scaled_image_final, (512, 512), interpolation=cv2.INTER_AREA)
    scaled_mask_foreground_final = cv2.resize(scaled_mask_foreground_final, (512, 512), interpolation=cv2.INTER_AREA)

    scaled_image_final = Image.fromarray(scaled_image_final.astype('uint8'), 'RGB')
    scaled_mask_foreground_final = Image.fromarray(scaled_mask_foreground_final.astype('uint8'), 'L')

    return scaled_image_final, scaled_mask_foreground_final

def infer(image, foreground_prompt, background_prompt, num_foreground_steps=20, num_background_steps=50, foreground_scale=0.5, translation_x=0.0, translation_y=0.0, rotation_angle=0, seed=0):
    # Process the image and mask
    print("here")
    mask_foreground = get_mask(Image.fromarray(image))
    mask_background = Image.fromarray(cv2.bitwise_not(np.array(mask_foreground)).astype('uint8'), 'L')

    scaled_image_final, scaled_mask_foreground = scale_image_and_masks(image, np.array(mask_foreground), np.array(mask_background), foreground_scale, (translation_x, translation_y), rotation_angle)
    scaled_mask_background = Image.fromarray(cv2.bitwise_not(np.array(scaled_mask_foreground)).astype('uint8'), 'L')

    # Generate the image
    generator = torch.manual_seed(seed)  # Seed for reproducibility
    canny_image = get_canny_image(scaled_image_final, 100, 200)

    foreground_image = foreground_pipe(
        foreground_prompt,
        num_inference_steps=num_foreground_steps,
        generator=generator,
        image=canny_image
    ).images[0]

    background_image = background_pipe(
        background_prompt,
        num_inference_steps=num_background_steps,
        generator=generator,
        image=foreground_image,
        mask_image=scaled_mask_background
    ).images[0]

    print("Canny shape: ", np.array(canny_image).shape)
    print("Mask shape: ", np.array(mask_foreground).shape, np.array(mask_background).shape)
    output_images = [canny_image, foreground_image, background_image, scaled_mask_foreground, scaled_mask_background]
    log_run(image, foreground_prompt, background_prompt, num_foreground_steps, num_background_steps, foreground_scale, translation_x, translation_y, rotation_angle, seed, output_images)
    return output_images

block = gr.Blocks().queue()

with block:
    with gr.Row():
        gr.Markdown("## Image Inpainting with ControlNet")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(source='upload', type="numpy")
            foreground_prompt = gr.Textbox(label="Foreground Prompt", value="A renaissance oil painting of a colorful, texturized chair")
            background_prompt = gr.Textbox(label="Background Prompt", value="movie theater Subjective perspective")
            generate_button = gr.Button(label="Generate")

            with gr.Accordion("Advanced options", open=False):
                foreground_steps = gr.Slider(1, 100, step=1, label="Foreground Num Inference Steps", value=50)
                background_steps = gr.Slider(1, 100, step=1, label="Background Num Inference Steps", value=50)
                foreground_scale = gr.Slider(0.0, 1.0, step=0.1, label="Foreground Scale", value=0.5)
                translation_x = gr.Slider(-0.5, 0.5, step=0.1, label="Translation X", value=0.0)
                translation_y = gr.Slider(-0.5, 0.5, step=0.1, label="Translation Y", value=0.0)
                rotation_angle = gr.Slider(0, 360, step=1, label="Rotation Angle", value=0)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=762040392)

        with gr.Column():
            output_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=3, height='auto')

    ips = [
        image_input,
        foreground_prompt,
        background_prompt,
        foreground_steps,
        background_steps,
        foreground_scale,
        translation_x,
        translation_y,
        rotation_angle,
        seed
    ]
    generate_button.click(fn=infer, inputs=ips, outputs=[output_gallery])

block.launch(server_name='0.0.0.0', share=True)