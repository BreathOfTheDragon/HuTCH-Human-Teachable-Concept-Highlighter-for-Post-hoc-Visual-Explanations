from concurrent.futures import ProcessPoolExecutor
import os
import cv2
from itertools import combinations
import shutil
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Directories
blackened_dir = "./blackened_saliency_images"
super_blackened_dir = "./super_blackened_saliency_images"

if os.path.exists(super_blackened_dir):
    shutil.rmtree(super_blackened_dir)
    
os.makedirs(super_blackened_dir, exist_ok=True)


def create_super_image(input_dir, output_dir, n):
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        ultra_counter = 0
        for value in range(1, n + 1):
            image_combinations = list(combinations(image_files, value))
            for combination in image_combinations:
                combined_image = None
                for img_file in combination:
                    img_path = os.path.join(input_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if combined_image is None:
                        combined_image = img.astype(np.float32)
                    else:
                        combined_image = np.maximum(combined_image, img)

                output_path = os.path.join(output_dir, f"{ultra_counter}.png")
                cv2.imwrite(output_path, combined_image.astype(np.uint8))
                ultra_counter += 1
    except Exception as e:
        print(f"Error in creating super images for {input_dir}: {e}")


def process_combinations(name_without_extension):
    specific_blackened_dir = os.path.join(blackened_dir, name_without_extension)

    specific_super_blackened_dir = os.path.join(super_blackened_dir, name_without_extension)

    create_super_image(specific_blackened_dir, specific_super_blackened_dir, 1)


if __name__ == "__main__":
    blackened_subdirs = [f for f in os.listdir(blackened_dir) if os.path.isdir(os.path.join(blackened_dir, f))]
    if not blackened_subdirs:
        raise FileNotFoundError(f"No subdirectories found in {blackened_dir}")

    with ProcessPoolExecutor() as executor:
        executor.map(process_combinations, blackened_subdirs)
