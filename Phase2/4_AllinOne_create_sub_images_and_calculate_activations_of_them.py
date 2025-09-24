import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import shutil
import torchvision
import cv2
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from collections import OrderedDict
import math
from itertools import combinations
import sys
import warnings
import os

warnings.filterwarnings("ignore")


images_to_highlight_for_computer_clean_dir = "./images_to_highlight_for_computer_clean"


images_to_highlight_for_computer_clean = [f for f in os.listdir(images_to_highlight_for_computer_clean_dir) if os.path.isfile(os.path.join(images_to_highlight_for_computer_clean_dir, f))]

print(images_to_highlight_for_computer_clean)



# Device Selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


# Data Preparation
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        return transforms.functional.pad(image, padding, 0, 'constant')



TRANSFORMS = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),    
    transforms.ToTensor(),  
    transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
])


# TRANSFORMS = transforms.Compose([
#     SquarePad(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# TRANSFORMS = transforms.Compose([
#     SquarePad(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])


class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if
                            f.lower().endswith(('jpg', 'jpeg', 'png')) and not f.startswith('.')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        image_name = self.image_files[idx]

        if self.transform:
            image = self.transform(image)

        return image, image_name


# Model Setup
LAYER_OF_INTEREST = 'layer4.2.conv2'
MODEL_PATH = './models/TransferModel_ResNet152_15_bee_wasp_only.pth'

model = models.resnet152(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

state_dict = torch.load(MODEL_PATH, map_location=device)
new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model = model.to(device)

# Activation Generator
class ActivationGenerator:
    def __init__(self, model, target_layer, device):
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.activations = {}

    def hook_layer(self):
        def hook(model, input, output):
            self.activations[self.target_layer] = output.detach()
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(hook)

    def get_activations(self, dataset_loader):
        self.model.eval()
        self.hook_layer()

        activations_dict = {}
        with torch.no_grad():
            for inputs, image_names in dataset_loader:
                inputs = inputs.to(self.device)
                _ = self.model(inputs)
                activations = self.activations[self.target_layer].cpu().numpy()
                for name, activation in zip(image_names, activations):
                    activations_dict[name] = activation

        return activations_dict

    def save_activations(self, activations, save_path):
        np.save(save_path, activations)
        print(f"Activations saved to {save_path}")



class Highlighter:
    def __init__(self):
        
        self.data_transforms = torchvision.transforms.Compose([
        # SquarePad(),   
        # transforms.Resize(224),  
        transforms.ToTensor(),  
        # transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
        ])

    def create_blackened_images_everything_else(self, input_image_path, output_dir, horizontal_step, vertical_step):
        # Load and transform the image
        image = Image.open(input_image_path).convert("RGBA")  # Ensure the image has an alpha channel
        image_tensor = self.data_transforms(image)

        # Remove old directory (if exists) and create a new one
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Dimensions of the transformed image
        h_limit, v_limit = image_tensor.shape[1], image_tensor.shape[2]

        index = 0
        for v_start in range(0, v_limit, vertical_step):
            for h_start in range(0, h_limit, horizontal_step):
                # Create a copy of the original tensor for RGBA
                rgba_tensor = torch.zeros((4, h_limit, v_limit))  # Initialize with fully transparent

                # Copy the patch of interest (RGB and Alpha)
                rgba_tensor[:3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step] = \
                    image_tensor[:3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step]  # RGB
                rgba_tensor[3, h_start:h_start + horizontal_step, v_start:v_start + vertical_step] = 1  # Alpha (opaque)

                # Convert tensor to a PIL image
                blackened_image = (rgba_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # Convert to numpy
                pil_image = Image.fromarray(blackened_image, mode="RGBA")

                # Save the resulting image
                pil_image.save(f"{output_dir}/{index}.png")
                index += 1



    def create_segmented_images(self, input_image_path, output_dir, threshold=0.2):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Load the pretrained Mask R-CNN model
        model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights="DEFAULT")
        model.eval()

        # Load and preprocess the image
        image = cv2.imread(input_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        pil_image = Image.fromarray(image_rgb)
        image_tensor = self.data_transforms(pil_image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)

        # Extract masks, boxes, and scores
        masks = outputs[0]['masks'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()

        # Set target resolution
        target_size = (224, 224)

        for i, score in enumerate(scores):
            if score < threshold:
                continue

            # Process the mask
            mask = (masks[i, 0] > 0.2).astype(np.uint8) * 255
            resized_mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

            # Apply the resized mask to the resized image
            segment = cv2.bitwise_and(resized_image, resized_image, mask=resized_mask)

            # Save the segmented region
            cv2.imwrite(f"{output_dir}/{i}.png", segment)

    def create_segmented_images_no_224_resize(self, input_image_path, output_dir, threshold):
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
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

        for i, score in enumerate(scores):
            if score < threshold:
                continue

            mask = (masks[i, 0] > 0.2).astype(np.uint8) * 255
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

            rgba_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            rgba_image[:, :, 3] = mask

            cv2.imwrite(f"{output_dir}/{i}.png", rgba_image)

    class SquarePad:
        def __call__(self, image):
            w, h = image.size
            max_wh = max(w, h)
            padding = [(max_wh - w) // 2, (max_wh - h) // 2]
            padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
            return transforms.functional.pad(image, padding, 0, 'constant')




highlighter = Highlighter()

blackened_dir = "./blackened_images"
segmented_dir = "./segmented_images"


if os.path.exists(blackened_dir):
    shutil.rmtree(blackened_dir)
    
if os.path.exists(segmented_dir):
    shutil.rmtree(segmented_dir)
    
for image in images_to_highlight_for_computer_clean:
    
    
    image_path = os.path.join(images_to_highlight_for_computer_clean_dir, image)
    the_image = Image.open(image_path)
    X, Y = the_image.size
    print(X , Y)
    X = math.floor(X/3)
    Y = math.floor(Y/3)
    print(X , Y)

    name_without_extension = os.path.splitext(image)[0]
    
    specific_blackened_dir = os.path.join(blackened_dir, name_without_extension)
    specific_segmented_dir = os.path.join(segmented_dir, name_without_extension)
    
    ############################################################################################################################
    # Create blackened images
    highlighter.create_blackened_images_everything_else(image_path, specific_blackened_dir, horizontal_step=X, vertical_step=Y)

    # Create segmented images
    highlighter.create_segmented_images_no_224_resize(image_path, specific_segmented_dir, threshold=0.0)
    ############################################################################################################################

    if os.path.exists(f'./super_blackened_images/{name_without_extension}'):
        shutil.rmtree(f'./super_blackened_images/{name_without_extension}')
    if os.path.exists(f'./super_segmented_images/{name_without_extension}'):
        shutil.rmtree(f'./super_segmented_images/{name_without_extension}')
        
    BASE_DIR = '.'
    specific_super_blackened_dir = os.path.join(BASE_DIR, 'super_blackened_images', name_without_extension)
    specific_super_segmented_dir = os.path.join(BASE_DIR, 'super_segmented_images', name_without_extension)

    if os.path.exists(specific_super_blackened_dir):
        shutil.rmtree(specific_super_blackened_dir)

    if os.path.exists(specific_super_segmented_dir):
        shutil.rmtree(specific_super_segmented_dir)

    os.makedirs(specific_super_blackened_dir, exist_ok=True)
    os.makedirs(specific_super_segmented_dir, exist_ok=True)


    def create_super_image(input_dir, output_dir, n):

        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('png', 'jpg', 'jpeg'))])

        ultra_counter = 0
        # Process each combination
        for value in range(1, n+1): # 1 to n inclusive
            image_combinations = list(combinations(image_files, value))
            for idx, combination in enumerate(image_combinations):
                combined_image = None
                for img_file in combination:
                    img_path = os.path.join(input_dir, img_file)
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)  # Load the image
                    if combined_image is None:
                        combined_image = img.astype(np.float32)  # Initialize combined image
                    else:
                        combined_image = np.maximum(combined_image, img)  # Combine by taking maximum pixel values

                # Save the combined image
                output_path = os.path.join(output_dir, f"{ultra_counter}.png")
                cv2.imwrite(output_path, combined_image.astype(np.uint8))  # Convert to uint8 for saving
                ultra_counter+=1

        print(f"Combined images saved to '{output_dir}' with n={n}.")




    create_super_image(specific_blackened_dir, specific_super_blackened_dir, 4)
    create_super_image(specific_segmented_dir, specific_super_segmented_dir, 4)



    ##############################################################################################
    #                          Generate activations for blackened images                         #
    ##############################################################################################


    #dataset = SingleClassDataset(blackened_dir, transform=TRANSFORMS)
    dataset = SingleClassDataset(specific_super_blackened_dir, transform=TRANSFORMS)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    activation_generator = ActivationGenerator(model, target_layer=LAYER_OF_INTEREST, device=device)

    activations_dict = activation_generator.get_activations(dataloader)

    temp_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"./name_activation_dict_blackened/{temp_name}_{LAYER_OF_INTEREST}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(activations_dict, f)

    print(f"Activations dictionary saved to {save_path}")


    ##############################################################################################
    #                          Generate activations for segmented images                         #
    ##############################################################################################


    #dataset = SingleClassDataset(segmented_dir, transform=TRANSFORMS)
    dataset = SingleClassDataset(specific_super_segmented_dir, transform=TRANSFORMS)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    activation_generator = ActivationGenerator(model, target_layer=LAYER_OF_INTEREST, device=device)

    activations_dict = activation_generator.get_activations(dataloader)


    temp_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = f"./name_activation_dict_segmented/{temp_name}_{LAYER_OF_INTEREST}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(activations_dict, f)

    print(f"Activations dictionary saved to {save_path}")


