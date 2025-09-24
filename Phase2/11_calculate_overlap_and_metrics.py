import os
import numpy as np
from PIL import Image
import pandas as pd

# Directories
computer_dir = "highlighted_saliency_images_by_computer"
expert_dir = "highlighted_images_by_expert"
overlap_dir = "highlighted_saliency_overlap"
os.makedirs(overlap_dir, exist_ok=True)

# Function to compute IoU and Dice
def compute_iou_dice(image1, image2):
    # Ensure both images are the same size
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

# Process each mode
results = []

computer_mode_dir = os.path.join(computer_dir)
expert_mode_dir = os.path.join(expert_dir)
overlap_mode_dir = os.path.join(overlap_dir)
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

    # Create an overlap mask to show only non-transparent parts of the expert image
    # that coincide with non-transparent parts of the saliency (computer) image
    computer_mask = computer_image_np[:, :, 3] > 0  # Non-transparent parts of the computer image
    expert_mask = expert_image_np[:, :, 3] > 0  # Non-transparent parts of the expert image

    overlap_mask = np.logical_and(computer_mask, expert_mask)
    result_image_np = np.zeros_like(expert_image_np)
    result_image_np[overlap_mask] = expert_image_np[overlap_mask]

    # Save the resulting image
    result_image = Image.fromarray(result_image_np, mode="RGBA")
    result_image.save(os.path.join(overlap_mode_dir, image_name))

    # Record results with values rounded to 3 decimals
    results.append({
        "image": image_name,
        "iou": round(iou, 3),
        "dice": round(dice, 3)
    })

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_csv_path = os.path.join(overlap_dir, "iou_dice_results_saliency.csv")
results_df.to_csv(results_csv_path, index=False)

print(f"Results saved to {results_csv_path}")
