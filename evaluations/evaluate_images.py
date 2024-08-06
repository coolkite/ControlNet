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

def process_image_pair(original_depth_path, compared_depth_path, compared_img_path, baseline_interval_path, learned_interval_path, output_suffix, mask, output_dir):
    """Process a pair of images (original and compared) and save results."""
    original_depth = load_image(original_depth_path, grayscale=True)
    compared_depth = load_image(compared_depth_path, grayscale=True)
    compared_img = load_image(compared_img_path)
    baseline_interval = load_image(baseline_interval_path)
    learned_interval = load_image(learned_interval_path)

    if mask is None or not isinstance(mask, np.ndarray):
        print(f"Warning: Invalid mask for {compared_img_path}. Skipping processing.")
        return

    # Resize mask to match the compared depth image size
    mask = cv2.resize(mask, (compared_depth.shape[1], compared_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply segmentation mask only to the compared depth image
    compared_depth_masked = apply_mask(compared_depth, mask)

    # Calculate MSE and IoU using original (unmasked) and compared (masked) depth images
    mse = calculate_mse(original_depth, compared_depth_masked)
    iou = calculate_iou(original_depth, compared_depth_masked)
    print(f"{output_suffix.capitalize()} MSE for {compared_img_path}: {mse:.2f}, IoU: {iou:.2f}")

    # Create combined image for visualization (original unmasked, compared masked)
    combined_img = create_combined_image(original_depth, compared_depth_masked, mse, iou)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save combined image
    output_path = os.path.join(output_dir, f"{output_suffix}_combined_masked.jpg")
    cv2.imwrite(output_path, combined_img)

    # Save masked depth image
    masked_depth_path = os.path.join(output_dir, f"{output_suffix}_depth_masked.jpg")
    cv2.imwrite(masked_depth_path, compared_depth_masked)

    # Save generated image
    generated_img_path = os.path.join(output_dir, f"{output_suffix}_generated.jpg")
    cv2.imwrite(generated_img_path, compared_img)

    # Save baseline interval image
    baseline_interval_path = os.path.join(output_dir, f"{output_suffix}_interval.jpg")
    cv2.imwrite(baseline_interval_path, baseline_interval)

    # Save learned interval image
    learned_interval_path = os.path.join(output_dir, f"{output_suffix}_learned_interval.jpg")
    cv2.imwrite(learned_interval_path, learned_interval)

    # Save metric
    with open(os.path.join(output_dir, f"{output_suffix}_metrics.txt"), 'w') as f:
        f.write(f"MSE: {mse:.2f}\nIoU: {iou:.2f}")


def apply_mask(image, mask):
    """Apply binary mask to the image."""
    if image.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Ensure the mask is binary
    _, mask = cv2.threshold(mask, 126, 255, cv2.THRESH_BINARY)
    
    # Convert mask to the same data type as the image
    mask = mask.astype(image.dtype)
    
    return cv2.bitwise_and(image, image, mask=mask)

def generate_mask_from_original_depth(original_depth_path):
    """Generate mask from original depth image."""
    original_depth = load_image(original_depth_path, grayscale=True)
    _, mask = cv2.threshold(original_depth, 1, 255, cv2.THRESH_BINARY)
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
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
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

# def process_images(input_data_dir, output_data_dir, sam_checkpoint_path):
#     """Process all images in the input directory and save results in the output directory."""
    
#     # Load the SAM model
#     sam_predictor = load_sam_model(sam_checkpoint_path)
#     for model_dir in os.listdir(input_data_dir):
#         model_path = os.path.join(input_data_dir, model_dir)
#         if not os.path.isdir(model_path):
#             continue

#         for shape_dir in os.listdir(model_path):
#             shape_path = os.path.join(model_path, shape_dir)
#             if not os.path.isdir(shape_path):
#                 continue

#             for interval_dir in os.listdir(shape_path):
#                 interval_path = os.path.join(shape_path, interval_dir)
#                 if not os.path.isdir(interval_path):
#                     continue

#                 for view_dir in os.listdir(interval_path):
#                     view_path = os.path.join(interval_path, view_dir)
#                     if not os.path.isdir(view_path):
#                         continue

#                     for seed_dir in os.listdir(view_path):
#                         seed_path = os.path.join(view_path, seed_dir)
#                         if not os.path.isdir(seed_path):
#                             continue

#                         # Process baseline and learned images
#                         baseline_path = None
#                         learned_path = None
#                         original_depth_path = None

#                         for file in os.listdir(seed_path):
#                             if file.endswith("_baseline.jpg"):
#                                 baseline_path = os.path.join(seed_path, file)
#                             elif file.endswith("_learned.jpg"):
#                                 learned_path = os.path.join(seed_path, file)
#                             elif file.endswith("_original_depth.jpg"):
#                                 original_depth_path = os.path.join(seed_path, file)

#                         if baseline_path and learned_path and original_depth_path:
#                             print(f"Generating masks for {baseline_path} and {learned_path}")
#                             # Generate masks using different methods
#                             mask_a = generate_mask_from_original_depth(original_depth_path)
#                             mask_b = generate_mask_from_union(baseline_path, learned_path)
#                             mask_c = generate_mask_from_dino(baseline_path)
#                             mask_d = generate_mask_from_dino(learned_path)

#                             # Generate masks using SAM
#                             mask_e = generate_mask_from_sam(baseline_path, sam_predictor)
#                             mask_f = generate_mask_from_sam(learned_path, sam_predictor)
                            
#                             for mask_method, mask in [('a_original_depth_mask', mask_a),
#                                                       ('b_union_dino_mask', mask_b),
#                                                       ('c_baseline_dino_mask', mask_c),
#                                                       ('d_learned_dino_mask', mask_d),
#                                                       ('e_baseline_sam_mask', mask_e),
#                                                       ('f_learned_sam_mask', mask_f)]:
#                                 if mask is not None:
#                                     print(f"Processing {mask_method} - Shape: {mask.shape}, Type: {mask.dtype}")
#                                     output_seed_dir = os.path.join(output_data_dir, mask_method, model_dir, shape_dir, interval_dir, view_dir, seed_dir)

#                                     # Process baseline
#                                     process_image_pair(
#                                         original_depth_path,
#                                         baseline_path.replace(".jpg", "_depth.jpg"),
#                                         baseline_path,
#                                         "baseline",
#                                         mask,
#                                         output_seed_dir
#                                     )

#                                     # Process learned
#                                     process_image_pair(
#                                         original_depth_path,
#                                         learned_path.replace(".jpg", "_depth.jpg"),
#                                         learned_path,
#                                         "learned",
#                                         mask,
#                                         output_seed_dir
#                                     )
#                                 else:
#                                     print(f"Skipping {mask_method} for {baseline_path} and {learned_path} due to invalid mask")

#                             print(f"Processed {baseline_path} and {learned_path}")
#                         else:
#                             print(f"Missing required files in {seed_path}")

def process_images(input_data_dir, output_data_dir, sam_checkpoint_path):
    """Process all images in the input directory and save results in the output directory."""  
    interval = input_data_dir.split("/")[-1]  
    for view_dir in os.listdir(input_data_dir):
        view_path = os.path.join(input_data_dir, view_dir)
        if not os.path.isdir(view_path):
            continue

        for seed_dir in os.listdir(view_path):
            seed_path = os.path.join(view_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue

            # Process baseline and learned images
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
                
                # Generate and process each mask sequentially
                mask_methods = [
                    ('a_original_depth_mask', generate_mask_from_original_depth, original_depth_path),
                    ('b_union_dino_mask', generate_mask_from_union, (baseline_path, learned_path)),
                    ('c_baseline_dino_mask', generate_mask_from_dino, baseline_path),
                    ('d_learned_dino_mask', generate_mask_from_dino, learned_path),
                    # ('e_baseline_sam_mask', generate_mask_from_sam, (baseline_path, sam_checkpoint_path)),
                    # ('f_learned_sam_mask', generate_mask_from_sam, (learned_path, sam_checkpoint_path))
                ]
                
                for mask_method, mask_func, mask_args in mask_methods:
                    print(f"Generating {mask_method}")
                    
                    if isinstance(mask_args, tuple):
                        mask = mask_func(*mask_args)
                    else:
                        mask = mask_func(mask_args)
                    
                    if mask is not None:
                        print(f"Processing {mask_method} - Shape: {mask.shape}, Type: {mask.dtype}")
                        output_seed_dir = os.path.join(output_data_dir, mask_method, "03001627_pointbert_100_controlnet", "03001627_13cdc9e018a811a3ad484915511ccff6", interval, view_dir, seed_dir)

                        # Process baseline
                        process_image_pair(
                            original_depth_path,
                            baseline_path.replace(".jpg", "_depth.jpg"),
                            baseline_path,
                            baseline_interval_path,
                            learned_interval_path,
                            "baseline",
                            mask,
                            output_seed_dir
                        )

                        # Process learned
                        process_image_pair(
                            original_depth_path,
                            learned_path.replace(".jpg", "_depth.jpg"),
                            learned_path,
                            baseline_interval_path,
                            learned_interval_path,
                            "learned",
                            mask,
                            output_seed_dir
                        )
                        
                        # Free up memory
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
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_data_dir, exist_ok=True)
    
    process_images(args.input_data_dir, args.output_data_dir, args.sam_checkpoint_path)
    
if __name__ == "__main__":
    main()