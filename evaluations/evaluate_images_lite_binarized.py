import argparse
import os
import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import sys
import shutil
import re
import sys
sys.path.append("segment-anything")
from segment_anything import sam_model_registry, SamPredictor

# Add DINO-ViT features to the path
sys.path.append('dino-vit-features')

from cosegmentation import find_cosegmentation

def load_image(path, grayscale=False):
    """Load an image from the given path."""
    if grayscale:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(path)

# def load_sam_model(checkpoint_path):
#     """Load the Segment Anything Model."""
#     sam = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
#     sam.to(device="cuda")
#     return SamPredictor(sam)
    
def calculate_mse(original, compared):
    """Calculate Mean Squared Error between two images without applying mask to original."""
    if original.shape != compared.shape:
        compared = cv2.resize(compared, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.mean((original.astype(float) - compared.astype(float)) ** 2)

def calculate_iou(original, compared):
    """Calculate Intersection over Union (IoU) between two images."""
    if original.shape != compared.shape:
        compared = cv2.resize(compared, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    intersection = np.logical_and(original, compared)
    union = np.logical_or(original, compared)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def create_combined_image(original, compared, mse, iou):
    """Create a combined image with MSE and IoU text."""
    if original.shape != compared.shape:
        compared = cv2.resize(compared, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
    combined = np.hstack((original, compared))
    cv2.putText(combined, f"MSE: {mse:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, f"IoU: {iou:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return combined

def apply_mask(image, mask):
    """Apply binary mask to the binary image."""
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Ensure the mask is binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Convert mask to the same data type as the image
    mask = mask.astype(image.dtype)
    
    return cv2.bitwise_and(image, mask)

def binarize_image(image, threshold):
    """Convert grayscale image to binary (0 or 255) using the specified threshold."""
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary

def process_image_pair(original_depth_path, compared_depth_path, compared_img_path, interval_path, output_suffix, mask, output_dir):
    """Process a pair of images (original and compared) and save results."""
    original_depth = load_image(original_depth_path, grayscale=True)
    compared_depth = load_image(compared_depth_path, grayscale=True)
    compared_img = load_image(compared_img_path)
    interval = load_image(interval_path)

    if mask is None or not isinstance(mask, np.ndarray):
        print(f"Warning: Invalid mask for {compared_img_path}. Skipping processing.")
        return

    # Binarize the original and compared depth images with different thresholds
    original_depth_binary = binarize_image(original_depth, threshold=0)
    compared_depth_binary = compared_depth
    #compared_depth_binary = binarize_image(compared_depth, threshold=100)

    # Resize mask to match the compared depth image size
    mask = cv2.resize(mask, (compared_depth_binary.shape[1], compared_depth_binary.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply segmentation mask to the binarized compared depth image
    compared_depth_masked_binary = apply_mask(compared_depth_binary, mask)

    # Calculate MSE and IoU using binarized images
    mse = calculate_mse(original_depth_binary, compared_depth_masked_binary)
    iou = calculate_iou(original_depth_binary, compared_depth_masked_binary)
    print(f"{output_suffix.capitalize()} MSE for {compared_img_path}: {mse:.2f}, IoU: {iou:.2f}")

    # Create combined image for visualization (original binarized, compared masked binarized)
    combined_img = create_combined_image(original_depth_binary, compared_depth_masked_binary, mse, iou)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined image
    output_path = os.path.join(output_dir, f"{output_suffix}_combined_masked_binary.jpg")
    cv2.imwrite(output_path, combined_img)

    # Save masked binary depth image
    masked_depth_path = os.path.join(output_dir, f"{output_suffix}_depth_masked_binary.jpg")
    cv2.imwrite(masked_depth_path, compared_depth_masked_binary)

    # Save generated image (unchanged)
    generated_img_path = os.path.join(output_dir, f"{output_suffix}_generated.jpg")
    cv2.imwrite(generated_img_path, compared_img)

    # Save interval image (unchanged)
    interval_path = os.path.join(output_dir, f"{output_suffix}_interval.jpg")
    cv2.imwrite(interval_path, interval)

    # Save metric
    with open(os.path.join(output_dir, f"{output_suffix}_metrics.txt"), 'w') as f:
        f.write(f"MSE: {mse:.2f}\nIoU: {iou:.2f}")

def generate_mask_from_original_depth(original_depth_path):
    """Generate mask from original depth image."""
    original_depth = load_image(original_depth_path, grayscale=True)
    _, mask = cv2.threshold(original_depth, 0, 255, cv2.THRESH_BINARY)
    return mask

def generate_mask_from_union(baseline_path, learned_path):
    """Generate mask from union of DINO masks."""
    seg_masks = perform_dino_vit_segmentation([baseline_path, learned_path])
    if seg_masks and len(seg_masks) == 2:
        union_mask = np.logical_or(seg_masks[0], seg_masks[1]).astype(np.uint8) * 255
        return union_mask
    else:
        print(f"Warning: DINO-ViT segmentation failed for union mask")
        return None

# def generate_mask_from_sam(image_path, sam_checkpoint_path):
#     """Generate mask using Segment Anything Model."""
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
#     sam_predictor = load_sam_model(sam_checkpoint_path)

#     # Set the image for the SAM predictor
#     sam_predictor.set_image(image)
    
#     # Predict masks with SAM
#     masks = sam_predictor.predict()
    
#     if len(masks) > 0:
#         # Select the mask with the largest area
#         largest_mask_index = np.argmax([np.sum(mask) for mask in masks])
#         mask = masks[largest_mask_index]
        
#         # Ensure the mask is 2D
#         if mask.ndim == 3:
#             mask = mask.squeeze()
        
#         # Convert mask to binary
#         mask = (mask > 0.5).astype(np.uint8) * 255
        
#         # Resize the mask to match the image size
#         mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
#         return mask
#     else:
#         print(f"Warning: SAM segmentation failed for {image_path}")
#         return None
    
def generate_mask_from_dino(image_path):
    """Generate mask using DINO-ViT segmentation."""
    seg_masks = perform_dino_vit_segmentation([image_path])
    if seg_masks and len(seg_masks) > 0:
        mask = seg_masks[0]
        
        # Convert to numpy array if it's not already
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        # Ensure the mask is 2D
        if mask.ndim == 3:
            mask = mask.squeeze()
        
        # Convert to uint8 if it's boolean or float
        if mask.dtype == bool:
            mask = mask.astype(np.uint8) * 255
        elif mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = (mask * 255).astype(np.uint8)
        
        # If the mask is not binary, threshold it
        if mask.max() > 1 or mask.min() < 0:
            _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
        
        print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
        
        return mask
    else:
        print(f"Warning: DINO-ViT segmentation failed for {image_path}")
        return None

def perform_dino_vit_segmentation(image_paths, load_size=375, layer=11, facet='key', bin=False, thresh=0.12,
                                  model_type='dino_vits8', stride=4, elbow=0.975, votes_percentage=75,
                                  remove_outliers=False, outliers_thresh=0.7, sample_interval=20,
                                  low_res_saliency_maps=True):
    """Perform DINO-ViT segmentation on the given images."""
    with torch.no_grad():
        seg_masks, _ = find_cosegmentation(image_paths, elbow, load_size, layer, facet, bin, thresh, model_type,
                                           stride, votes_percentage, sample_interval, remove_outliers,
                                           outliers_thresh, low_res_saliency_maps)
    torch.cuda.empty_cache()
    return seg_masks

def process_images(shape, interval, view, seed, input_data_dir, output_data_dir, sam_predictor):
    """Process images for a specific combination of shape, interval, view, and seed."""
    model_dir = "03001627_pointbert_100_controlnet"
    shape_dir = f"03001627_{shape}"
    interval_dir = str(interval)
    view_dir = str(view)
    seed_dir = str(seed)

    seed_path = os.path.join(input_data_dir, model_dir, shape_dir, interval_dir, view_dir, seed_dir)

    if not os.path.isdir(seed_path):
        print(f"Directory not found: {seed_path}")
        return

    baseline_path = None
    learned_path = None
    original_depth_path = None
    baseline_interval_path = None
    learned_interval_path = None

    for file in os.listdir(seed_path):
        if file.endswith("_baseline.jpg"):
            baseline_path = os.path.join(seed_path, file)
        elif file.endswith("_learned.jpg"):
            learned_path = os.path.join(seed_path, file)
        elif file.endswith("_original_depth.jpg"):
            original_depth_path = os.path.join(seed_path, file)
        elif file.endswith("_baseline_inter.jpg"):
            baseline_interval_path = os.path.join(seed_path, file)
        elif file.endswith("_learned_inter.jpg"):
            learned_interval_path = os.path.join(seed_path, file)

    if baseline_path and learned_path and original_depth_path and baseline_interval_path and learned_interval_path:
        print(f"Processing {baseline_path} and {learned_path}")
        
        mask_methods = [
            ('a_original_depth_mask', generate_mask_from_original_depth, original_depth_path),
            ('c_baseline_dino_mask', generate_mask_from_dino, baseline_path),
            ('d_learned_dino_mask', generate_mask_from_dino, learned_path),
        ]
        #('b_union_dino_mask', generate_mask_from_union, (baseline_path, learned_path)),
        
        for mask_method, mask_func, mask_args in mask_methods:
            print(f"Generating {mask_method}")
            
            if isinstance(mask_args, tuple):
                mask = mask_func(*mask_args)
            else:
                mask = mask_func(mask_args)
            
            if mask is not None:
                print(f"Processing {mask_method} - Shape: {mask.shape}, Type: {mask.dtype}")
                output_seed_dir = os.path.join(output_data_dir, mask_method, model_dir, shape_dir, interval_dir, view_dir, seed_dir)

                process_image_pair(
                    original_depth_path,
                    baseline_path.replace(".jpg", "_depth.jpg"),
                    baseline_path,
                    baseline_interval_path,
                    "baseline",
                    mask,
                    output_seed_dir
                )

                process_image_pair(
                    original_depth_path,
                    learned_path.replace(".jpg", "_depth.jpg"),
                    learned_path,
                    learned_interval_path,
                    "learned",
                    mask,
                    output_seed_dir
                )
                
                del mask
                torch.cuda.empty_cache()
            else:
                print(f"Skipping {mask_method} for {baseline_path} and {learned_path} due to invalid mask")

        print(f"Processed {baseline_path} and {learned_path}")
    else:
        print(f"Missing required files in {seed_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str, required=True, help="Path to the directory containing the input images")
    parser.add_argument("--output_data_dir", type=str, required=True, help="Path to saving directory for processed images and outputs")
    parser.add_argument("--sam_checkpoint_path", type=str, required=True, help="Path to the SAM checkpoint")
    parser.add_argument("--shape", type=str, required=True, help="Shape ID")
    parser.add_argument("--interval", type=int, required=True, help="Interval value")
    parser.add_argument("--view", type=int, required=True, help="View index")
    parser.add_argument("--seed", type=int, required=True, help="Seed value")
    args = parser.parse_args()
    
    os.makedirs(args.output_data_dir, exist_ok=True)
    
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint_path)
    sam.to(device="cuda")
    sam_predictor = SamPredictor(sam)
    
    process_images(args.shape, args.interval, args.view, args.seed, args.input_data_dir, args.output_data_dir, sam_predictor)
    
if __name__ == "__main__":
    main()