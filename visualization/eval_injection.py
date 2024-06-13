import sys
#sys.path.append('/project/pi_ekalogerakis_umass_edu/dshivashok/geometry-editing-2d3d/')
from diffusion_sd import StableDiffusionPipelineNormal
from clip_functions import get_clip_embeddings

from typing import List, Optional
import numpy as np
import torch
from tqdm import tqdm
import os
import argparse
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="Token Visualization")
    parser.add_argument("--seed", type=int, default=8888)
    parser.add_argument("--prompt", type=str, default="a 3d render of a TOKEN")
    parser.add_argument("--num_diffusion_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="zz_srun_debug")
    parser.add_argument("--token_path", type=str, default="zz_srun_debug")
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-2-1')
    parser.add_argument("--torch_seed", type=int, default=8888)

    return parser.parse_args()

def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def latent2image(vae, latents):
    '''
    convert a latent vector repr into image using VAE
    '''
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    '''
    initialize a latent vector
    '''
    if latent is None:
        latent = torch.randn(
            (1, model.unet.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

############################

@torch.no_grad()
def text2image_ldm_stable_modified(
    model,
    prompt: List[str],
    args,
    num_inference_steps: int = 50,
    generator: Optional[torch.Generator] = None,
    guidance_scale: float = 7.5,
):
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    token_ids = text_input.input_ids.to(model.device)
    print(token_ids)
    inputs_embeds = model.text_encoder.text_model.embeddings.token_embedding(token_ids)
    np.save("inputs_embeds.npy", inputs_embeds.cpu().detach().numpy())
    decoded_tokens = model.tokenizer.decode(token_ids[0])
    print("Decoded Tokens:", decoded_tokens)

    token = torch.from_numpy(np.load(args.token_path))
    inputs_embeds, embed, weights = operate_on_embed(model, inputs_embeds, token, idx=[7], type="chair-four+three")
    print("Embed shape: ", inputs_embeds.shape)
    print(inputs_embeds[:,7,:])
    text_embeddings = get_clip_embeddings(model.text_encoder.text_model, token_ids, inputs_embeds=inputs_embeds)[0]
    print("Text embeddings: ", text_embeddings.shape)
    print(text_embeddings[:,7,:])
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]

    context = [uncond_embeddings, text_embeddings]
    context = torch.cat(context)
    latent, latents = init_latent(None, model, height, width, generator, batch_size)

    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, latents, context, t, guidance_scale, False)

    image = latent2image(model.vae, latents)

    return image, latent, embed, weights

def operate_on_embed(model, inputs_embeds, noise, idx=[0], type="add"):
    embeds = noise
    chair_weight, three_weight = [None, None]
    if type == "identity":
        inputs_embeds[0,idx[0],:] = inputs_embeds[0,idx[0],:]
    if type == "add":
        inputs_embeds[0,idx[0],:] = inputs_embeds[0,idx[0],:] + noise
    if type == "replace":
        print("noise shape: ", noise.shape)
        inputs_embeds[0,idx[0],:] = noise
    if type == "chair_three_avg":
        text_input = model.tokenizer(
            "chair three",
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = text_input.input_ids.to(model.device)
        print(token_ids)
        temp_embeds = model.text_encoder.text_model.embeddings.token_embedding(token_ids)
        chair_weight, three_weight = [0.5, 0.5]
        embeds = temp_embeds[0,1,:]*chair_weight + temp_embeds[0,2,:]*three_weight
        inputs_embeds[0,idx[0],:] = embeds
    if type == "chair-four+three":
        text_input = model.tokenizer(
            "chair four three",
            padding="max_length",
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        token_ids = text_input.input_ids.to(model.device)
        print(token_ids)
        temp_embeds = model.text_encoder.text_model.embeddings.token_embedding(token_ids)
        #chair_weight, three_weight = [0.5, 0.5]
        embeds = temp_embeds[0,1,:] - 1.5*temp_embeds[0,2,:] + 1.5*temp_embeds[0,3,:]
        inputs_embeds[0,idx[0],:] = embeds
    return inputs_embeds, embeds, [chair_weight, three_weight]

def run_and_display_modified(model, prompts, args, num_inference_steps=50, generator=None, file_name="image", seed=None):
    TORCH_SEED = seed
    torch.manual_seed(TORCH_SEED)

    print("Torch seed is: ", TORCH_SEED)
    images, x_t, embeds, weights = text2image_ldm_stable_modified(model, prompts, args, num_inference_steps=num_inference_steps, generator=generator, guidance_scale=7)
    images_fin = save_images(images, file_name+f"_chair_{weights[0]}_three_{weights[1]}")
    np.save(f"{file_name}_embeds_chair_{weights[0]}_three_{weights[1]}.npy", embeds.cpu().detach().numpy())
    print("Done")
    return images, x_t, embeds

def save_images(images, file_name="cross_attention", num_rows=1, offset_ratio=0.02):
    '''
    display a collection of images in a grid format
    '''
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    print(file_name)

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    pil_img.save(f"{file_name}.png")

def main():
    args = get_args()
    print(args)
    ldm_stable = StableDiffusionPipelineNormal(args.model).model
    g_cpu = torch.Generator().manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    name = f"seed_{args.seed}_torch_seed_{args.torch_seed}_num_steps_{args.num_diffusion_steps}"
    #args.token_path[:-4].split('/')[-1] + '_vis_'

    images, latents, inputs_embeds = run_and_display_modified(ldm_stable,
                                                              [args.prompt],
                                                              args,
                                                              num_inference_steps=args.num_diffusion_steps,
                                                              generator=g_cpu,
                                                              file_name=os.path.join(args.output_dir, name),
                                                              seed=args.torch_seed,
                                                              )

if __name__ == '__main__':
    main()