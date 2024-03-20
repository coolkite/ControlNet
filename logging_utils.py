import json
import datetime
import numpy as np
import base64
from PIL import Image
from io import BytesIO

def log_data(image_input, foreground_prompt, background_prompt, foreground_steps, background_steps, foreground_scale, seed, output_gallery):
    print(output_gallery)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"timestamp": timestamp}
    
    # Convert input image to base64 string
    buffered = BytesIO()
    Image.fromarray(image_input).save(buffered, format="PNG")
    image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data["image_input"] = image_str
    
    # Store other input parameters
    data["foreground_prompt"] = foreground_prompt
    data["background_prompt"] = background_prompt
    data["foreground_steps"] = foreground_steps
    data["background_steps"] = background_steps
    data["foreground_scale"] = foreground_scale
    data["seed"] = seed
    
    # Store output gallery images as base64 strings
    output_images = []
    for image in output_gallery:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        output_images.append(image_str)
    print("output gallery", len(output_gallery))
    print("output images", len(output_images))


    data["output_gallery"] = output_images

    # Generate HTML content
    html_content = "<html><body>"
    html_content += f"<h2>Experiment Log - {timestamp}</h2>"
    html_content += "<table>"
    html_content += f"<tr><td>Input Image</td><td><img src='data:image/png;base64,{data['image_input']}' width='200' height='200'/></td></tr>"
    html_content += f"<tr><td>Foreground Prompt</td><td>{data['foreground_prompt']}</td></tr>"
    html_content += f"<tr><td>Background Prompt</td><td>{data['background_prompt']}</td></tr>"
    html_content += f"<tr><td>Foreground Steps</td><td>{data['foreground_steps']}</td></tr>"
    html_content += f"<tr><td>Background Steps</td><td>{data['background_steps']}</td></tr>"
    html_content += f"<tr><td>Foreground Scale</td><td>{data['foreground_scale']}</td></tr>"
    html_content += f"<tr><td>Seed</td><td>{data['seed']}</td></tr>"
    html_content += "<tr><td>Output Gallery</td><td>"
    for image_str in data["output_gallery"]:
        html_content += f"<img src='data:image/png;base64,{image_str}' width='200' height='200'/>"
    html_content += "</td></tr>"
    html_content += "</table>"
    html_content += "</body></html>"

    with open("experiment_log.html", "a") as file:
        file.write(html_content)

def update_experiment_log():
    try:
        with open("experiment_log.html", "r") as file:
            html_content = file.read()
    except FileNotFoundError:
        html_content = "No experiments logged yet."
    return html_content