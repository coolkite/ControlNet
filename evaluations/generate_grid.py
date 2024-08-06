import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.utils import ImageReader
import io

def resize_image(img, target_height):
    aspect_ratio = img.width / img.height
    new_height = target_height
    new_width = int(new_height * aspect_ratio)
    return img.resize((new_width, new_height), Image.LANCZOS)

def generate_grid_images(base_dir, output_dir):
    for model_dir in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_dir)
        if not os.path.isdir(model_path):
            continue

        for shape_dir in os.listdir(model_path):
            shape_path = os.path.join(model_path, shape_dir)
            if not os.path.isdir(shape_path):
                continue

            intervals = sorted([d for d in os.listdir(shape_path) if os.path.isdir(os.path.join(shape_path, d))], key=int)
            
            # Create a PDF for this model and shape
            pdf_filename = f"{model_dir}_{shape_dir}_grids.pdf"
            pdf_path = os.path.join(output_dir, model_dir, shape_dir, pdf_filename)
            c = canvas.Canvas(pdf_path, pagesize=landscape(A4))

            for view_dir in os.listdir(os.path.join(shape_path, intervals[0])):
                view_path = os.path.join(shape_path, intervals[0], view_dir)
                if not os.path.isdir(view_path):
                    continue

                for seed_dir in os.listdir(view_path):
                    seed_path = os.path.join(view_path, seed_dir)
                    if not os.path.isdir(seed_path):
                        continue

                    # Set image size and spacing
                    img_height = 256
                    spacing = 20
                    header_height = 40
                    label_width = 100
                    
                    # Calculate grid dimensions after resizing images
                    sample_interval = intervals[0]
                    sample_path = os.path.join(shape_path, sample_interval, view_dir, seed_dir)
                    sample_image = Image.open(os.path.join(sample_path, "baseline_generated.jpg"))
                    sample_aspect_ratio = sample_image.width / sample_image.height
                    img_width = int(img_height * sample_aspect_ratio)

                    grid_width = (img_width * 9 + spacing * 6) + label_width
                    grid_height = (img_height + spacing) * len(intervals) + header_height + spacing

                    grid = Image.new('RGB', (grid_width, grid_height), color='white')
                    draw = ImageDraw.Draw(grid)
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)

                    # Add header
                    headers = ["Baseline MSE", "Baseline", "Baseline Interval", "Learned MSE", "Learned", "Learned Interval"]
                    for i, header in enumerate(headers):
                        x = int(label_width + spacing + (1.5 * i * (img_width + spacing)) + img_width // 2)
                        draw.text((x, 20), header, font=font, fill="black", anchor="ms")

                    for i, interval in enumerate(intervals):
                        y_offset = i * (img_height + spacing) + header_height + spacing
                        
                        # Add interval label
                        draw.text((5, y_offset + img_height // 2), f"Interval: {interval}", font=font, fill="black", anchor="lm")

                        interval_path = os.path.join(shape_path, interval, view_dir, seed_dir)
                        
                        baseline_mse_path = os.path.join(interval_path, "baseline_combined_masked.jpg")
                        baseline_path = os.path.join(interval_path, "baseline_generated.jpg")
                        baseline_interval_path = os.path.join(interval_path, "baseline_interval.jpg")
                        learned_mse_path = os.path.join(interval_path, "learned_combined_masked.jpg")
                        learned_path = os.path.join(interval_path, "learned_generated.jpg")
                        learned_interval_path = os.path.join(interval_path, "learned_interval.jpg")

                        if all(os.path.isfile(path) for path in [baseline_mse_path, baseline_path, baseline_interval_path, learned_mse_path, learned_path, learned_interval_path]):
                            images = [
                                resize_image(Image.open(baseline_mse_path), img_height),
                                resize_image(Image.open(baseline_path), img_height),
                                resize_image(Image.open(baseline_interval_path), img_height),
                                resize_image(Image.open(learned_mse_path), img_height),
                                resize_image(Image.open(learned_path), img_height),
                                resize_image(Image.open(learned_interval_path), img_height),
                            ]
                            
                            x_offset = label_width + spacing
                            for img in images:
                                grid.paste(img, (x_offset, y_offset))
                                x_offset += img.width + spacing
                        else:
                            print(f"Missing images for interval {interval}, view {view_dir}, seed {seed_dir}")

                    # Create output directory structure
                    output_subdir = os.path.join(output_dir, model_dir, shape_dir)
                    os.makedirs(output_subdir, exist_ok=True)

                    # Save the grid
                    grid_filename = f"{model_dir}_{shape_dir}_view_{view_dir}_seed_{seed_dir}_grid.png"
                    grid_path = os.path.join(output_subdir, grid_filename)
                    grid.save(grid_path, "PNG", quality=95, dpi=(300, 300))
                    print(f"Saved grid: {grid_filename}")

                    # Add the grid to the PDF
                    img_reader = ImageReader(grid_path)
                    c.setPageSize((grid_width, grid_height))
                    c.drawImage(img_reader, 0, 0, width=grid_width, height=grid_height)
                    c.showPage()

            # Save and close the PDF
            c.save()
            print(f"Saved PDF: {pdf_filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate image grids from processed images.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the base directory containing processed images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the generated grids")
    args = parser.parse_args()

    generate_grid_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()