import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import time
import warnings
from PIL import Image
from datetime import datetime
import torchvision.transforms.functional as F
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
warnings.filterwarnings("ignore")


# Device setup
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


# Class for padding images to make them square
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        return transforms.functional.pad(image, padding, 0, 'constant')

# Data transformations


data_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),    
    transforms.ToTensor(),  
    transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
])



data_transforms_for_saliency = transforms.Compose([
    SquarePad(),
    transforms.Resize(224)
])

# Function to compute saliency map
def compute_saliency_map(model, input_tensor, threshold):
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_()
    model.to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    saliency = input_tensor.grad.data.abs()
    saliency, _ = saliency.max(dim=1)
    saliency = saliency.squeeze().cpu().numpy()

    saliency[saliency < threshold] = 0

    return saliency

# Function to save saliency map as an RGB image
def save_saliency_map_RGB(saliency_map, save_path):
    saliency_map_normalized = (saliency_map / saliency_map.max() * 255).astype(np.uint8)
    rgb_image = np.stack([saliency_map_normalized] * 3, axis=-1)
    Image.fromarray(rgb_image, mode="RGB").save(save_path)

# Function to save saliency map as an RGBA image
def save_saliency_map_RGBA(saliency_map, save_path):
    saliency_map_normalized = (saliency_map / saliency_map.max() * 255).astype(np.uint8)
    rgba_image = np.zeros((saliency_map.shape[0], saliency_map.shape[1], 4), dtype=np.uint8)
    rgba_image[:, :, :3] = np.stack([saliency_map_normalized] * 3, axis=-1)
    rgba_image[:, :, 3] = saliency_map_normalized
    Image.fromarray(rgba_image, mode="RGBA").save(save_path)

# Function to plot and save the side-by-side saliency map
def plot_saliency_map(original_image, saliency_map, save_path):
    original_image_transformed = data_transforms_for_saliency(original_image)
    
    saliency_map_normalized = (saliency_map / saliency_map.max() * 255).astype(np.uint8)

    # saliency_map_normalized = (saliency_map * 255).astype(np.uint8)

    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(original_image_transformed)
    ax[0].axis('off')
    ax[0].set_title('Original Image')

    ax[1].imshow(saliency_map_normalized, cmap='hot')
    ax[1].axis('off')
    ax[1].set_title('Saliency Map')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

# Load the model
model = models.resnet152(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)

BASE_DIR = '.'
MODEL_PATH = os.path.join(BASE_DIR, 'models/TransferModel_ResNet152_15_bee_wasp_only.pth')

state_dict = torch.load(MODEL_PATH, map_location=device)
new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model.to(device)

# Directory for saliency outputs
saliency_dir = "saliency_images_by_model"
os.makedirs(saliency_dir, exist_ok=True)
os.makedirs(os.path.join(saliency_dir, "side_by_side"), exist_ok=True)
os.makedirs(os.path.join(saliency_dir, "RGB"), exist_ok=True)
os.makedirs(os.path.join(saliency_dir, "RGBA"), exist_ok=True)

# Directory with input images
images_to_highlight_dir = "./images_to_highlight_for_expert_clean"
images_to_highlight = [
    f for f in os.listdir(images_to_highlight_dir) if os.path.isfile(os.path.join(images_to_highlight_dir, f))
]

# Generate saliency maps for each image
for image_name in images_to_highlight:
    image_path = os.path.join(images_to_highlight_dir, image_name)
    image = Image.open(image_path).convert('RGB')

    input_tensor = data_transforms(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(device))

    predicted_class = output.argmax(dim=1).item()

    saliency_map = compute_saliency_map(model, input_tensor, threshold= 0.05)

    # Save paths
    rgb_save_path = os.path.join(saliency_dir, "RGB", f"{image_name}")
    rgba_save_path = os.path.join(saliency_dir, "RGBA", f"{image_name}")
    side_by_side_save_path = os.path.join(saliency_dir, "side_by_side", f"{image_name}")

    # Save saliency maps
    save_saliency_map_RGB(saliency_map, rgb_save_path)
    save_saliency_map_RGBA(saliency_map, rgba_save_path)
    plot_saliency_map(image, saliency_map, side_by_side_save_path)
