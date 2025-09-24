import os
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from collections import OrderedDict
import warnings
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



warnings.filterwarnings("ignore")

# Device Selection
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")



# Directories for organizing the project
BASE_DIR = '.'
CONCEPTS_DIR = os.path.join(BASE_DIR, 'AugmentedConceptsFinal')  # Directory containing concept examples
ACTIVATIONS_DIR = os.path.join(BASE_DIR, 'activations')  # Directory to save activations
CAVS_DIR = os.path.join(BASE_DIR, 'cavs')  # Directory to save CAVs
MODEL_PATH = os.path.join(BASE_DIR, 'models/TransferModel_ResNet152_15_bee_wasp_only.pth')  # Path to the pretrained model

# Ensure necessary directories exist
os.makedirs(ACTIVATIONS_DIR, exist_ok=True)
os.makedirs(CAVS_DIR, exist_ok=True)





# LAYER_OF_INTEREST = 'layer4.2.conv2'  # The layer in the model to extract activations from

LAYER_OF_INTEREST = 'avgpool'  




# CONCEPTS = ["BentAntenna", "Fur", "LongAntenna", "PinchWaist", "SegmentedAntenna", "Stripes"]  # List of concepts

# CONCEPTS = [ "Fur", "PinchWaist", "Stripes"]

CONCEPTS = [ "Fur", "PinchWaist"]


# Define a padding class to preprocess images
class SquarePad:
    def __call__(self, image):
        w, h = image.size  # Get image dimensions
        max_wh = max(w, h)  # Determine the largest dimension
        # Calculate padding for width and height
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        return transforms.functional.pad(image, padding, 0, 'constant')

# Custom dataset to load images
class SingleClassDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir  # Directory containing images
        self.transform = transform  # Transformations to apply to the images
        # List all valid image files
        self.image_files = [f for f in os.listdir(root_dir) if
                            f.lower().endswith(('jpg', 'jpeg', 'png')) and not f.startswith('.')]

    def __len__(self):
        return len(self.image_files)  # Return the number of images

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])  # Get image path
        image = Image.open(img_path).convert("RGB")  # Load and convert to RGB
        if self.transform:
            image = self.transform(image)  # Apply transformations
        return image, 0  # Return image and dummy label (0)

# Define transformations to apply to images

TRANSFORMS = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),    
    transforms.ToTensor(),  
    transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
])



# TRANSFORMS = transforms.Compose([
#     SquarePad(),  # Pad image to make it square
#     transforms.Resize((224, 224)),  # Resize to 224x224
#     transforms.ToTensor(),  # Convert to tensor
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ImageNet stats
# ])




# TRANSFORMS = transforms.Compose([
#     SquarePad(),  # Pad image to make it square
#     transforms.Resize((224, 224)),  # Resize to 224x224
#     transforms.ToTensor()  # Convert to tensor
# ])

# Model Setup

model = models.resnet152(pretrained=True)  # Load pretrained ResNet-152 model
model.fc = nn.Linear(model.fc.in_features, 2)  # Replace the fully connected layer


state_dict = torch.load(MODEL_PATH, map_location=device)

new_state_dict = OrderedDict((k.replace('module.', ''), v) for k, v in state_dict.items())
model.load_state_dict(new_state_dict) 
model = model.to(device)  


class ActivationGenerator:
    def __init__(self, model, target_layer, device):
        self.model = model  
        self.target_layer = target_layer  
        self.device = device  
        self.activations = {}  
        self.counter = 0

    def hook_layer(self):
        # Define the hook function to capture the activations
        def hook(model, input, output):
            self.activations[self.target_layer] = output.detach()
            # Print the shape of the activations each time the hook is called
            print(f"[Hook] Activation from layer '{self.target_layer}' has shape: {output.shape}")
        # Register the hook on the desired layer
        layer = dict(self.model.named_modules())[self.target_layer]
        layer.register_forward_hook(hook)

    def get_activations(self, dataloader):
        self.model.eval() 
        self.hook_layer()  

        all_activations = []
        with torch.no_grad():
            for inputs, _ in dataloader:
                inputs = inputs.to(self.device)  
                _ = self.model(inputs)  
                batch_activation = self.activations[self.target_layer].cpu().numpy()
                # Print the shape for the current batch
                print(f"[Batch] Captured activation shape: {batch_activation.shape}")
                all_activations.append(batch_activation)
                
                self.counter += 1
                print(self.counter)
                
        activations = np.concatenate(all_activations, axis=0)  
        print(f"[Total] The size of the corresponding activations is: {activations.shape}")

        return activations

    @staticmethod
    def save_activations(activations, save_path):
        np.save(save_path, activations)  # Save activations as a .npy file
        print(f"Activations saved to {save_path}")


# Class to train and save Concept Activation Vectors (CAVs)

class CAV:
    def __init__(self, positive_activations, negative_activations, model_layer, concept_name, save_path):
        self.positive_activations = positive_activations  
        self.negative_activations = negative_activations  
        self.model_layer = model_layer  
        self.concept_name = concept_name  
        self.save_path = save_path 
        self.scaler = StandardScaler()  
        self.cav = None 
        self.intercept = None  

    def train(self):
       
        positive_flat = self.positive_activations.reshape(len(self.positive_activations), -1)
        negative_flat = self.negative_activations.reshape(len(self.negative_activations), -1)

        # Combine positive and negative activations
        X = np.concatenate([positive_flat, negative_flat], axis=0)
        y = np.concatenate([np.ones(len(positive_flat)), -np.ones(len(negative_flat))], axis=0)


        X = self.scaler.fit_transform(X)

        # Train a linear classifier (SVC with linear kernel)
        classifier = LogisticRegression(
            penalty='l2',         
            C=1.0,                
            solver='lbfgs',        
            max_iter=10000,         
            verbose=1,             
            random_state=42        
        )
        classifier.fit(X, y)

        # Store CAV coefficients and intercept
        self.cav = classifier.coef_
        self.intercept = classifier.intercept_
        self.save()

    def save(self):
        os.makedirs(self.save_path, exist_ok=True)  # Ensure save directory exists
        # Save CAV components
        np.save(os.path.join(self.save_path, f"CAV_{self.concept_name}_{self.model_layer}.npy"), self.cav)
        np.save(os.path.join(self.save_path, f"Intercept_{self.concept_name}_{self.model_layer}.npy"), self.intercept)
        np.save(os.path.join(self.save_path, f"Scaler_{self.concept_name}_{self.model_layer}.npy"), self.scaler)


# Process each concept directory
for concept_dir in ["Negative", "Positive"]:
    concept_path = os.path.join(CONCEPTS_DIR, concept_dir)
    for concept in os.listdir(concept_path):
        if not os.path.isdir(os.path.join(concept_path, concept)):
            continue  

        dataset = SingleClassDataset(os.path.join(concept_path, concept), transform=TRANSFORMS)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)


        activation_gen = ActivationGenerator(model, target_layer=LAYER_OF_INTEREST, device=device)
        activations = activation_gen.get_activations(dataloader)

        ActivationGenerator.save_activations(activations, f"{ACTIVATIONS_DIR}/{concept}_{LAYER_OF_INTEREST}.npy")


for concept in CONCEPTS:
   
    positive_activations = np.load(f"{ACTIVATIONS_DIR}/Positive{concept}_{LAYER_OF_INTEREST}.npy")
    negative_activations = np.load(f"{ACTIVATIONS_DIR}/Negative{concept}_{LAYER_OF_INTEREST}.npy")


    cav = CAV(positive_activations, negative_activations, model_layer=LAYER_OF_INTEREST, concept_name=concept, save_path=CAVS_DIR)
    cav.train()
