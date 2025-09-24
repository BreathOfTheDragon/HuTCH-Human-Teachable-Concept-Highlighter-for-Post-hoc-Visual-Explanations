import os
import numpy as np
from PIL import Image
import pandas as pd

# Directories
computer_dir = "highlighted_images_by_computer"
expert_dir = "highlighted_images_by_expert"
overlap_dir = "highlighted_images_overlap"
os.makedirs(overlap_dir, exist_ok=True)

# Subdirectories
modes = ["blackened", "segmented"]

# Function to compute IoU and Dice
def compute_iou_dice(image1, image2):
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError(f"Image sizes do not match: {image1.shape[:2]} vs {image2.shape[:2]}")

    # Convert images to binary masks
    mask1 = np.array(image1[:, :, 3] > 0, dtype=np.uint8)  # Use alpha channel
    mask2 = np.array(image2[:, :, 3] > 0, dtype=np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0

    return iou, dice


for mode in modes:
    results = []
    computer_mode_dir = os.path.join(computer_dir, mode)
    expert_mode_dir = os.path.join(expert_dir)
    overlap_mode_dir = os.path.join(overlap_dir, mode)
    os.makedirs(overlap_mode_dir, exist_ok=True)

    # List of image names (assuming filenames match in both directories)
    image_names = [f for f in os.listdir(computer_mode_dir) if f.endswith(".png")]

    for image_name in image_names:
        computer_image_path = os.path.join(computer_mode_dir, image_name)
        expert_image_path = os.path.join(expert_mode_dir, image_name)

        # Open images
        computer_image = Image.open(computer_image_path).convert("RGBA")
        expert_image = Image.open(expert_image_path).convert("RGBA")

        # Resize images to match dimensions if necessary
        if computer_image.size != expert_image.size:
            print(f"Resizing {image_name} as sizes do not match: {computer_image.size} vs {expert_image.size}")
            expert_image = expert_image.resize(computer_image.size, Image.Resampling.LANCZOS)

        # Convert to numpy arrays
        computer_image_np = np.array(computer_image)
        expert_image_np = np.array(expert_image)

        # Compute IoU and Dice
        iou, dice = compute_iou_dice(computer_image_np, expert_image_np)

        # Intersection: Keep only the common areas (alpha intersection)
        intersection_alpha = np.minimum(computer_image_np[:, :, 3], expert_image_np[:, :, 3])
        intersection_rgb = np.zeros_like(computer_image_np[:, :, :3], dtype=np.float32)

        for c in range(3):  # For each color channel
            intersection_rgb[:, :, c] = (
                computer_image_np[:, :, c] * (intersection_alpha / 255.0) +
                expert_image_np[:, :, c] * (intersection_alpha / 255.0)
            ) / 2  # Average RGB values in the common area

        # Combine RGB and alpha
        intersection_rgb = np.clip(intersection_rgb, 0, 255).astype(np.uint8)
        intersection_image = np.dstack((intersection_rgb, intersection_alpha))

        # Save the intersection image
        result_image = Image.fromarray(intersection_image.astype(np.uint8), mode="RGBA")
        result_image.save(os.path.join(overlap_mode_dir, image_name))

        # Record results with values rounded to 4 decimals
        results.append({
            "mode": mode,
            "image": image_name,
            "iou": round(iou, 3),
            "dice": round(dice, 3)
        })

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(overlap_dir, f"iou_dice_results_{mode}.csv")
    results_df.to_csv(results_csv_path, index=False)

    print(f"Results saved to {results_csv_path}")
