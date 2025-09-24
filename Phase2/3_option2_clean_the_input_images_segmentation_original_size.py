import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import os
import torchvision
import shutil
import warnings
from PIL import Image

warnings.filterwarnings("ignore")

types_of_audience = ["computer", "expert"]

computer_result_dir = "./images_to_highlight_for_computer_clean"
if os.path.exists(computer_result_dir):
    shutil.rmtree(computer_result_dir)
os.makedirs(computer_result_dir, exist_ok=True)

expert_result_dir = "./images_to_highlight_for_expert_clean"
if os.path.exists(expert_result_dir):
    shutil.rmtree(expert_result_dir)
os.makedirs(expert_result_dir, exist_ok=True)


model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=True)
model.eval()

images_to_highlight_unclean_dir = "./images_to_highlight_unclean"
images_to_highlight_unclean = [f for f in os.listdir(images_to_highlight_unclean_dir) if os.path.isfile(os.path.join(images_to_highlight_unclean_dir, f))]

for image in images_to_highlight_unclean:
    name_without_extension = os.path.splitext(image)[0]

    image_path = os.path.join(images_to_highlight_unclean_dir, image)
    image = cv2.imread(image_path)
    print(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 


    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)


    with torch.no_grad():
        outputs = model(image_tensor)

  
    masks = outputs[0]['masks'].cpu().numpy()  # Masks
    scores = outputs[0]['scores'].cpu().numpy()

    for audience in types_of_audience:
        result_dir = computer_result_dir if audience == "computer" else expert_result_dir
        # Important: 
        threshold = 0.3  

        # Filter masks based on confidence
        for i, score in enumerate(scores):
            if score < threshold:
                continue

            # Process the mask
            mask_threshold = 0.05 if audience == "computer" else 0
            mask = (masks[i, 0] > mask_threshold).astype(np.uint8) * 255

            # Find the bounding box of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue  

            # Get the bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            # Crop the mask and the original image to the bounding box
            cropped_mask = mask[y:y + h, x:x + w]
            cropped_segment = image[y:y + h, x:x + w]

            # Create a transparent RGBA image
            cropped_segment_rgba = cv2.cvtColor(cropped_segment, cv2.COLOR_BGR2RGBA)
            alpha_channel = cropped_mask  # Use the mask as the alpha channel
            cropped_segment_rgba[:, :, 3] = alpha_channel


            output_path = os.path.join(result_dir, f"{name_without_extension}.png")
            Image.fromarray(cropped_segment_rgba).save(output_path)

            print(f"Mask R-CNN cropped segmentation saved in '{result_dir}' for '{audience}' audience")
            break