import os
import numpy as np
from PIL import Image
import shutil

# Directories
base_dir = "./super_blackened_saliency_images"  # Each subdirectory corresponds to one image's regions
output_dir = "./highlighted_saliency_images_by_computer"

# Remove and recreate the output directory if it exists
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Number of top regions to select and combine per image (subdirectory)




no_of_top_regions = 1





# Function to calculate the average value for the non-transparent parts of an image
def calculate_average_non_transparent(image_path):
    img = Image.open(image_path).convert("RGBA")
    img_array = np.array(img)
    
    # Extract the alpha channel
    alpha_channel = img_array[:, :, 3]
    
    # Create a mask for non-transparent pixels (alpha > 0)
    non_transparent_mask = alpha_channel > 0
    
    # If there are any non-transparent pixels, compute the mean of their RGB values
    if non_transparent_mask.any():
        rgb_values = img_array[:, :, :3][non_transparent_mask]
        avg_value = np.mean(rgb_values)
        return avg_value
    else:
        return 0  # Fully transparent image

# Function to combine multiple images using alpha blending
def combine_images(image_paths):
    # Open all images and ensure they are in RGBA mode
    images = [Image.open(img_path).convert("RGBA") for img_path in image_paths]
    
    # Optional: Ensure all images have the same size. Here, we use the size of the first image.
    width, height = images[0].size
    images = [img.resize((width, height)) for img in images]
    
    # Convert images to numpy arrays (float32 for precision during blending)
    image_arrays = [np.array(img, dtype=np.float32) for img in images]
    
    # Initialize arrays to accumulate the weighted RGB values and sum of alpha values
    combined_array = np.zeros_like(image_arrays[0], dtype=np.float32)
    alpha_sum = np.zeros((height, width), dtype=np.float32)
    
    # Blend each image according to its alpha channel
    for img_array in image_arrays:
        # Normalize the alpha channel to [0, 1]
        alpha = img_array[:, :, 3] / 255.0
        # Weight the RGB channels by the alpha value
        combined_array[:, :, :3] += img_array[:, :, :3] * alpha[..., None]
        # Accumulate the alpha values
        alpha_sum += alpha
    
    # Avoid division by zero: for pixels where alpha_sum is positive, divide to get the weighted average
    combined_array[:, :, :3] = np.divide(
        combined_array[:, :, :3],
        alpha_sum[..., None],
        out=np.zeros_like(combined_array[:, :, :3]),
        where=alpha_sum[..., None] > 0
    )
    # Restore the alpha channel (scale back to [0, 255])
    combined_array[:, :, 3] = np.clip(alpha_sum * 255, 0, 255)
    
    # Convert the result back to an unsigned 8-bit integer array and create a PIL image
    final_combined_array = combined_array.astype(np.uint8)
    combined_image = Image.fromarray(final_combined_array, mode="RGBA")
    return combined_image

# Process each subdirectory in the base directory
for sub_dir in os.listdir(base_dir):
    sub_dir_path = os.path.join(base_dir, sub_dir)
    if not os.path.isdir(sub_dir_path):
        continue

    print(f"Processing directory: {sub_dir}")
    
    # Gather (filename, average value) pairs for each image in the subdirectory
    image_scores = []
    for img_file in os.listdir(sub_dir_path):
        if img_file.lower().endswith(('png', 'jpg', 'jpeg')):
            img_path = os.path.join(sub_dir_path, img_file)
            avg_value = calculate_average_non_transparent(img_path)
            image_scores.append((img_file, avg_value))
    
    # Sort the images by average value (highest first)
    image_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select the top no_of_top_regions images
    top_images = image_scores[:no_of_top_regions]
    print(f"Top {no_of_top_regions} images in {sub_dir}:")
    for fname, score in top_images:
        print(f"  {fname}: {score:.2f}")
    
    # Build full paths for the selected images
    top_image_paths = [os.path.join(sub_dir_path, fname) for fname, _ in top_images]
    
    # Combine the selected images using alpha blending
    combined_image = combine_images(top_image_paths)
    
    # Save the composite image to the output directory with a name based on the subdirectory name
    output_path = os.path.join(output_dir, f"{sub_dir}.png")
    combined_image.save(output_path)
    print(f"Saved combined image to: {output_path}\n")
