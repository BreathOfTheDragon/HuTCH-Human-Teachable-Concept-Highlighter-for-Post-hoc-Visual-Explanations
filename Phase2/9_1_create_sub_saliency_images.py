from concurrent.futures import ProcessPoolExecutor
import os
import math
import torch
import torchvision
from PIL import Image
import warnings
import shutil
import numpy as np

warnings.filterwarnings("ignore")

# Directories


saliency_images_by_model_dir = "./saliency_images_by_model/RGB"

# saliency_images_by_model_dir = "./saliency_images_by_model/RGBA"



blackened_dir = "./blackened_saliency_images"

# Ensure output directories are clean
if os.path.exists(blackened_dir):
    shutil.rmtree(blackened_dir)

os.makedirs(blackened_dir, exist_ok=True)

# Get list of images
saliency_images_by_model = [
    f for f in os.listdir(saliency_images_by_model_dir)
    if os.path.isfile(os.path.join(saliency_images_by_model_dir, f))
]

if not saliency_images_by_model:
    raise FileNotFoundError(f"No images found in {saliency_images_by_model_dir}")

print(f"Images to process: {saliency_images_by_model}")


class Highlighter:
    def __init__(self):
        self.data_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])

    def create_blackened_sub_images(self, input_image_path, output_dir, horizontal_step, vertical_step):
        os.makedirs(output_dir, exist_ok=True)

        image = Image.open(input_image_path).convert("RGBA")
        image_tensor = self.data_transforms(image)

        index = 1
        h_limit, v_limit = image_tensor.shape[1], image_tensor.shape[2]

        for v_start in range(0, v_limit, vertical_step):
            for h_start in range(0, h_limit, horizontal_step):
                rgba_tensor = torch.zeros((4, h_limit, v_limit))
                rgba_tensor[:3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step] = \
                    image_tensor[:3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step]
                rgba_tensor[3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step] = \
                    image_tensor[3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step]

                blackened_image = (rgba_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(blackened_image, mode="RGBA")

                file_path = os.path.join(output_dir, f"{index}_{horizontal_step}_{vertical_step}.png")
                pil_image.save(file_path)
                index += 1


highlighter = Highlighter()


def process_image(image, lower_div, upper_div):
    image_path = os.path.join(saliency_images_by_model_dir, image)
    name_without_extension = os.path.splitext(image)[0]

    specific_blackened_dir = os.path.join(blackened_dir, name_without_extension)
    os.makedirs(specific_blackened_dir, exist_ok=True)

    the_image = Image.open(image_path)
    X, Y = the_image.size

    for div1 in np.arange(lower_div, upper_div + 1, 1):
        X_step = max(1, math.floor(float(X) / div1))
        for div2 in np.arange(lower_div, upper_div + 1, 1):
            Y_step = max(1, math.floor(float(Y) / div2))
            highlighter.create_blackened_sub_images(
                image_path, specific_blackened_dir, horizontal_step=X_step, vertical_step=Y_step
            )


if __name__ == "__main__":
    from functools import partial

    lower_div = 2
    upper_div = 4

    with ProcessPoolExecutor() as executor:
        process_image_partial = partial(process_image, lower_div=lower_div, upper_div=upper_div)
        executor.map(process_image_partial, saliency_images_by_model)
