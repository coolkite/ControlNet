import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def process_depth_files(base_dir):
    results = []
    
    # Process all .npz files in the directory
    for file in tqdm(os.listdir(base_dir)):
        if file.endswith('.npz'):
            file_path = os.path.join(base_dir, file)
            chair_id = os.path.splitext(file)[0]  # Get chair ID from filename
            
            depth_maps = np.load(file_path)["arr_0"]
            print(depth_maps.shape)
            if depth_maps.shape[0] != 20:
                print(f"Warning: Unexpected number of views in {file}. Expected 20, got {depth_maps.shape[0]}")
                continue
            
            # Flatten all dimensions except the first (views)
            flattened_depths = depth_maps.reshape(20, -1)
            
            # Compute mean and std dev across all pixels in all views for this chair
            mean_depth = np.mean(flattened_depths)
            std_depth = np.std(flattened_depths)
            
            results.append({
                'chair_id': chair_id,
                'mean': mean_depth,
                'std': std_depth
            })
    
    return results

def main():
    base_dir = '/work/pi_ekalogerakis_umass_edu/dmpetrov/data/token_geometry/shapenet_pointbert_1000/depth_images'  # Replace with your actual path
    output_csv = 'chair_depth_statistics.csv'
    
    print("Processing depth files...")
    results = process_depth_files(base_dir)
    
    print("Creating CSV file...")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()