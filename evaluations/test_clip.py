from diffusers import StableDiffusionPipeline

# Load the Stable Diffusion 2.1 model
model_id = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# Access the model configuration
model_config = pipeline.config

# Print the model configuration details
print("Model Configuration:")
print(model_config)

# Check the URL for the text encoder and text model
text_encoder_config = pipeline.text_encoder.config
clip_model_id = text_encoder_config._name_or_path

print(f"\nCLIP Text Encoder Model ID: {clip_model_id}")
