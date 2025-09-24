from concurrent.futures import ProcessPoolExecutor
import os
import math
import torch
import torchvision
from PIL import Image
import cv2
import warnings
import shutil
import numpy as np
warnings.filterwarnings("ignore")


images_to_highlight_for_computer_clean_dir = "./images_to_highlight_for_computer_clean"
blackened_dir = "./blackened_images"
segmented_dir = "./segmented_images"

if os.path.exists(blackened_dir):
    shutil.rmtree(blackened_dir)

if os.path.exists(segmented_dir):
    shutil.rmtree(segmented_dir)
    
os.makedirs(blackened_dir, exist_ok=True)
os.makedirs(segmented_dir, exist_ok=True)


images_to_highlight_for_computer_clean = [
    f for f in os.listdir(images_to_highlight_for_computer_clean_dir)
    if os.path.isfile(os.path.join(images_to_highlight_for_computer_clean_dir, f))
]

if not images_to_highlight_for_computer_clean:
    raise FileNotFoundError(f"No images found in {images_to_highlight_for_computer_clean_dir}")

print(f"Images to process: {images_to_highlight_for_computer_clean}")

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
            



    def create_segmented_sub_images(self, input_image_path, output_dir, threshold):
    
        os.makedirs(output_dir, exist_ok=True)

        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        model.eval()

        image = cv2.imread(input_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_tensor = self.data_transforms(pil_image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)

        masks = outputs[0]['masks'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        for index, score in enumerate(scores):
            if score < threshold:
                continue

            mask = (masks[index, 0] > 0.2).astype(np.uint8) * 255
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba_image[:, :, 3] = mask

            file_path = f"{output_dir}/mask_{index}.png"
            cv2.imwrite(file_path, rgba_image)


highlighter = Highlighter()

def process_image(image, lower_div, upper_div):
    image_path = os.path.join(images_to_highlight_for_computer_clean_dir, image)
    name_without_extension = os.path.splitext(image)[0]

    specific_blackened_dir = os.path.join(blackened_dir, name_without_extension)
    specific_segmented_dir = os.path.join(segmented_dir, name_without_extension)

    os.makedirs(specific_blackened_dir, exist_ok=True)
    os.makedirs(specific_segmented_dir, exist_ok=True)

    
    the_image = Image.open(image_path)
    X, Y = the_image.size

    for div1 in np.arange(lower_div, upper_div +1, 1):
        
        X_step = max(1, math.floor(float(X) / div1))
        
        for div2 in np.arange(lower_div, upper_div +1, 1):
            
            Y_step = max(1, math.floor(float(Y) / div2))            
            highlighter.create_blackened_sub_images(
                image_path, specific_blackened_dir, horizontal_step=X_step, vertical_step=Y_step
            )

    highlighter.create_segmented_sub_images(
        image_path, specific_segmented_dir, threshold=0.1
    )
        






if __name__ == "__main__":
    from functools import partial
    import numpy as np


    lower_div = 2
    upper_div = 4


    with ProcessPoolExecutor() as executor:
        process_image_partial = partial(process_image, lower_div=lower_div, upper_div=upper_div)

        executor.map(process_image_partial, images_to_highlight_for_computer_clean)
