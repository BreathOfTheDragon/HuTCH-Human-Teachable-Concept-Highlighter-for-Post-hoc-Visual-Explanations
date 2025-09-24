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
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")

# Define a custom transform to pad the image to a square
class SquarePad:
    def __call__(self, image):
        w, h = image.size  # PIL image: (width, height)
        max_wh = max(w, h)
        # Calculate padding: left, top, right, bottom
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

# Compose the transforms: first pad the image to a square, then resize to 224x224.
TRANSFORMS = transforms.Compose([
    SquarePad(),
    transforms.Resize(224)  # When passed an int, Resize resizes the smaller edge to that value.
])

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
images_to_highlight_unclean = [
    f for f in os.listdir(images_to_highlight_unclean_dir)
    if os.path.isfile(os.path.join(images_to_highlight_unclean_dir, f))
]

for image in images_to_highlight_unclean:
    name_without_extension = os.path.splitext(image)[0]
    image_path = os.path.join(images_to_highlight_unclean_dir, image)
    print(image_path)

    # Read the image using OpenCV and convert it to RGB
    image_cv = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    # Convert to tensor for the model
    image_tensor = F.to_tensor(image_rgb).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)

    masks = outputs[0]['masks'].cpu().numpy()  # Shape: [N, 1, H, W]
    scores = outputs[0]['scores'].cpu().numpy()

    for audience in types_of_audience:
        result_dir = computer_result_dir if audience == "computer" else expert_result_dir
        threshold = 0.3  # Confidence threshold

        # Process masks based on confidence
        for i, score in enumerate(scores):
            if score < threshold:
                continue

            # Set mask threshold based on audience type
            mask_threshold = 0.05 if audience == "computer" else 0
            mask = (masks[i, 0] > mask_threshold).astype(np.uint8) * 255

            # Find contours to get the bounding box of the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            # Use the largest contour to determine the bounding box
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            # Crop the mask and the original image to the bounding box
            cropped_mask = mask[y:y + h, x:x + w]
            cropped_segment = image_cv[y:y + h, x:x + w]

            # Create a transparent RGBA image by combining the cropped image and the mask
            cropped_segment_rgba = cv2.cvtColor(cropped_segment, cv2.COLOR_BGR2RGBA)
            cropped_segment_rgba[:, :, 3] = cropped_mask  # Set alpha channel

            # Convert the NumPy array to a PIL Image
            output_image = Image.fromarray(cropped_segment_rgba, mode="RGBA")
            # Apply the composed transforms: SquarePad and Resize(224)
            transformed_image = TRANSFORMS(output_image)

            output_path = os.path.join(result_dir, f"{name_without_extension}.png")
            transformed_image.save(output_path)

            print(f"Mask R-CNN cropped segmentation saved in '{result_dir}' for '{audience}' audience")
            # Process only the first valid mask for this audience
            break
