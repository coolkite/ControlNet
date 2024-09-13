import os
import shutil
import argparse

def extract_depth_images(input_dir, output_dir):
    for model_id in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_id)
        if os.path.isdir(model_path):
            # Extract category_id from the model_id
            category_id = model_id.split('_')[0]
            
            # Create corresponding directories in the output folder
            output_model_path = os.path.join(output_dir, category_id, model_id)
            os.makedirs(output_model_path, exist_ok=True)
            
            # Copy depth images
            for file in os.listdir(model_path):
                if file.endswith('_depth0001.png'):
                    src_file = os.path.join(model_path, file)
                    dst_file = os.path.join(output_model_path, file)
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract depth images from the original dataset")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the original dataset directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for depth images")
    
    args = parser.parse_args()
    
    extract_depth_images(args.input_dir, args.output_dir)
    print("Depth image extraction complete.")