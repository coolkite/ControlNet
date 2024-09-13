import argparse
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def load_image(path, grayscale=False):
    """Load an image from the given path."""
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)

def create_mask(depth_image):
    """Create a mask for non-zero depth values."""
    return depth_image > 0

def process_images(shape, interval, view, seed, input_data_dir, output_data_dir):
    """Process images for a specific combination of shape, interval, view, and seed."""
    model_dir = "03001627_pointbert_100_controlnet"
    shape_dir = shape
    interval_dir = str(interval)
    view_dir = str(view)
    seed_dir = str(seed)

    seed_path = os.path.join(input_data_dir, model_dir, shape_dir, interval_dir, view_dir, seed_dir)

    if not os.path.isdir(seed_path):
        print(f"Directory not found: {seed_path}")
        return

    original_depth_path = os.path.join(seed_path, f"{shape}_view_{view}_seed_{seed}_interval_{interval}_original_depth.jpg")
    learned_depth_path = os.path.join(seed_path, f"{shape}_view_{view}_seed_{seed}_interval_{interval}_learned_depth.jpg")
    baseline_depth_path = os.path.join(seed_path, f"{shape}_view_{view}_seed_{seed}_interval_{interval}_baseline_depth.jpg")

    print(original_depth_path)
    print(learned_depth_path)
    print(baseline_depth_path)

    if os.path.exists(original_depth_path) and os.path.exists(learned_depth_path) and os.path.exists(baseline_depth_path):
        original_depth = load_image(original_depth_path, grayscale=True)
        learned_depth = load_image(learned_depth_path, grayscale=True)
        baseline_depth = load_image(baseline_depth_path, grayscale=True)

        orig_mask = create_mask(original_depth)
        learned_mask = create_mask(learned_depth)
        baseline_mask = create_mask(baseline_depth)

        plt.figure(figsize=(6, 6))
        a = plt.hist(original_depth[orig_mask].reshape(-1), bins=50, color='blue', alpha=0.7, label='gt depth')
        plt.hist(learned_depth[learned_mask].reshape(-1), bins=a[1], color='red', alpha=0.7, label='learned token')
        plt.hist(baseline_depth[baseline_mask].reshape(-1), bins=a[1], color='green', alpha=0.7, label='baseline token')
        plt.legend()
        plt.title('Non-zero depth histograms')

        output_dir = os.path.join(output_data_dir, model_dir, shape_dir, interval_dir, view_dir, seed_dir)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'depth_hist.pdf'))
        plt.close()

        print(f"Processed and saved depth histogram for {seed_path}")
    else:
        print(f"Missing required files in {seed_path}")

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--input_data_dir", type=str, required=True, help="Path to the directory containing the input images")
    # parser.add_argument("--output_data_dir", type=str, required=True, help="Path to saving directory for processed images and outputs")
    # parser.add_argument("--shape", type=str, required=True, help="Shape ID")
    # parser.add_argument("--interval", type=int, required=True, help="Interval value")
    # parser.add_argument("--view", type=int, required=True, help="View index")
    # parser.add_argument("--seed", type=int, required=True, help="Seed value")
    # args = parser.parse_args()
    
    # os.makedirs(args.output_data_dir, exist_ok=True)
    
    # process_images(args.shape, args.interval, args.view, args.seed, args.input_data_dir, args.output_data_dir)
    input_data_dir = "/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/data/sd_1_5"
    output_data_dir = "/project/pi_ekalogerakis_umass_edu/dshivashok/ControlNet/evaluations/plots/sd_1_5"
    # shape = "03001627_1bec15f362b641ca7350b1b2f753f3a2"
    # interval = 25
    # view = 0
    # seed = 0
    # os.makedirs(output_data_dir, exist_ok=True)
    # process_images(shape, interval, view, seed, input_data_dir, output_data_dir)
    shapes = os.listdir(os.path.join(input_data_dir, '03001627_pointbert_100_controlnet'))
    for shape in shapes:
        shape_dir = os.path.join(input_data_dir, '03001627_pointbert_100_controlnet', shape)
        intervals = os.listdir(shape_dir)
        for interval in intervals:
            interval_dir = os.path.join(shape_dir, interval)
            views = os.listdir(interval_dir)
            for view in views:
                view_dir = os.path.join(interval_dir, view)
                seeds = os.listdir(view_dir)
                for seed in seeds:
                    process_images(shape, int(interval), int(view), int(seed), input_data_dir, output_data_dir)

if __name__ == "__main__":
    main()