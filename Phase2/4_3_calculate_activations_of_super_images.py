import os
import numpy as np
import torch
import torch.nn as nn
import pickle
import torchvision
import cv2
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from collections import OrderedDict
import warnings

warnings.filterwarnings("ignore")

images_to_highlight_for_computer_clean_dir = "./images_to_highlight_for_computer_clean"

images_to_highlight_for_computer_clean = [
    f for f in os.listdir(images_to_highlight_for_computer_clean_dir)
    if os.path.isfile(os.path.join(images_to_highlight_for_computer_clean_dir, f))
]

print(images_to_highlight_for_computer_clean)

# Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data Preparation
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        return transforms.functional.pad(image, padding, 0, 'constant')

# TRANSFORMS = transforms.Compose([
#     SquarePad(),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])


TRANSFORMS = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),    
    transforms.ToTensor(),  
    transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
])




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
        image = Image.open(img_path).convert("RGBA")  # Keep alpha channel
        image_name = self.image_files[idx]

        # Separate RGB and alpha channels
        rgba_image = np.array(image)
        alpha_channel = rgba_image[:, :, 3]

        # Check if the image is fully transparent
        if np.all(alpha_channel == 0):  # Fully transparent
            print(f"Skipping fully transparent image: {img_path}")
            return None, None

        # Find the bounding box of visible (non-transparent) areas
        non_zero_indices = np.argwhere(alpha_channel > 0)
        top_left = non_zero_indices.min(axis=0)
        bottom_right = non_zero_indices.max(axis=0) + 1
        cropped_rgba_image = rgba_image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]

        # Remove the alpha channel for model processing
        cropped_rgb_image = cropped_rgba_image[:, :, :3]

        # Convert back to PIL Image for transformations
        cropped_image = Image.fromarray(cropped_rgb_image)

        if self.transform:
            cropped_image = self.transform(cropped_image)

        return cropped_image, image_name


# Custom collate function to filter out None entries
def custom_collate(batch):
    batch = [item for item in batch if item[0] is not None]  # Filter out None
    if not batch:
        return None  # Return None if the entire batch is invalid
    images, image_names = zip(*batch)
    return torch.stack(images), image_names

# Model Setup

#LAYER_OF_INTEREST = 'layer4.2.conv2'

LAYER_OF_INTEREST = 'avgpool'  


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

    def get_activations(self, dataset_loader, save_images_dir=None, num_images_to_save=5):
        self.model.eval()
        self.hook_layer()

        activations_dict = {}
        saved_images_count = 0

        with torch.no_grad():
            for batch in dataset_loader:
                if batch is None:  # Skip empty batches
                    continue
                inputs, image_names = batch
                inputs = inputs.to(self.device)

                # Save the first few images if save_images_dir is specified
                if save_images_dir and saved_images_count < num_images_to_save:
                    for img_tensor, name in zip(inputs, image_names):
                        save_path = os.path.join(save_images_dir, f"saved_{name}")
                        os.makedirs(save_images_dir, exist_ok=True)
                        img_to_save = F.to_pil_image(img_tensor.cpu())
                        img_to_save.save(save_path)
                        saved_images_count += 1
                        if saved_images_count >= num_images_to_save:
                            break

                # Perform forward pass and store activations
                _ = self.model(inputs)
                activations = self.activations[self.target_layer].cpu().numpy()
                for name, activation in zip(image_names, activations):
                    activations_dict[name] = activation

        return activations_dict

# Main Loop
blackened_dir = "./blackened_images"
segmented_dir = "./segmented_images"
SAVE_IMAGES_DIR_BLACKENED = "./debug_images/blackened"
SAVE_IMAGES_DIR_SEGMENTED = "./debug_images/segmented"

for image in images_to_highlight_for_computer_clean:
    image_path = os.path.join(images_to_highlight_for_computer_clean_dir, image)
    name_without_extension = os.path.splitext(image)[0]

    specific_blackened_dir = os.path.join(blackened_dir, name_without_extension)
    specific_segmented_dir = os.path.join(segmented_dir, name_without_extension)

    specific_super_blackened_dir = os.path.join("super_blackened_images", name_without_extension)
    specific_super_segmented_dir = os.path.join("super_segmented_images", name_without_extension)

    ##############################################################################################
    # Generate activations for blackened images
    ##############################################################################################
    dataset = SingleClassDataset(specific_super_blackened_dir, transform=TRANSFORMS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    activation_generator = ActivationGenerator(model, target_layer=LAYER_OF_INTEREST, device=device)

    activations_dict = activation_generator.get_activations(
        dataloader, save_images_dir=SAVE_IMAGES_DIR_BLACKENED, num_images_to_save=5
    )

    save_path = f"./name_activation_dict_blackened/{name_without_extension}_{LAYER_OF_INTEREST}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(activations_dict, f)

    print(f"Activations dictionary saved to {save_path}")

    ##############################################################################################
    # Generate activations for segmented images
    ##############################################################################################
    dataset = SingleClassDataset(specific_super_segmented_dir, transform=TRANSFORMS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    activation_generator = ActivationGenerator(model, target_layer=LAYER_OF_INTEREST, device=device)

    activations_dict = activation_generator.get_activations(
        dataloader, save_images_dir=SAVE_IMAGES_DIR_SEGMENTED, num_images_to_save=5
    )

    save_path = f"./name_activation_dict_segmented/{name_without_extension}_{LAYER_OF_INTEREST}.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(activations_dict, f)

    print(f"Activations dictionary saved to {save_path}")
