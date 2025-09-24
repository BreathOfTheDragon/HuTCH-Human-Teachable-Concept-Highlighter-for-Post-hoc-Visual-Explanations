import torch
import torch.nn as nn
from PIL import Image
import os
import numpy as np
from collections import OrderedDict
import pickle
from torchvision import datasets, models, transforms, io
import shutil
import torchvision
from torch.utils.data import DataLoader, Dataset
import warnings
import torchvision.transforms.functional as F

warnings.filterwarnings("ignore")
torch.set_printoptions(threshold=float('inf'), linewidth=20000)




# Mode can be "segmented" or "blackened"
modes = ["blackened", "segmented"]

final_result_save_dir = "highlighted_images_by_computer"
if os.path.exists(final_result_save_dir):
    shutil.rmtree(final_result_save_dir)
os.makedirs(final_result_save_dir, exist_ok=True)



class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        image = F.pad(image, padding)
        return F.resize(image, (224, 224))


# test_transforms = transforms.Compose([
#     SquarePad(),
#     transforms.Resize(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])



test_transforms = transforms.Compose([
    SquarePad(),
    transforms.Resize(224),    
    transforms.ToTensor(),  
    transforms.Normalize([0.511, 0.495, 0.359], [0.206, 0.193, 0.190])  
])


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Device: {device}")


model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)

model = model.to(device)
model.load_state_dict(torch.load(f'./models/TransferModel_ResNet152_15_bee_wasp_only.pth'))
model.eval()


for mode in modes:
    
    
    
    
    no_of_top_regions = 3
    
    
    
    
    
    images_to_highlight_for_computer_clean_dir = "./images_to_highlight_for_computer_clean"
    images_to_highlight_for_computer_clean = [
        f for f in os.listdir(images_to_highlight_for_computer_clean_dir) 
        if os.path.isfile(os.path.join(images_to_highlight_for_computer_clean_dir, f))
    ]

    def flatten_activations(activations):
        return activations.reshape(1, -1)

    def compute_dot_CAV_score(concept, layer, modified_image_activation):
        cav = np.load(f"./cavs/CAV_{concept}_{layer}.npy")
        intercept = np.load(f"./cavs/Intercept_{concept}_{layer}.npy")
        scaler = np.load(f"./cavs/Scaler_{concept}_{layer}.npy", allow_pickle=True).item()

        modified_image_activation_flat = flatten_activations(modified_image_activation)
        modified_image_activation_flat = scaler.transform(modified_image_activation_flat)
        dot_value = np.dot(modified_image_activation_flat, cav.T) + intercept

        return dot_value

    Bee_Wasp_MEAN = [0.511, 0.495, 0.359]
    Bee_Wasp_STD = [0.206, 0.193, 0.190]
    
    
        
    
    
    #LAYER_OF_INTEREST = 'layer4.2.conv2'
    
    LAYER_OF_INTEREST = 'avgpool'  


    for image in images_to_highlight_for_computer_clean:
        image_path = os.path.join(images_to_highlight_for_computer_clean_dir, image)
        name_without_extension = os.path.splitext(image)[0]
        print(name_without_extension)

        if mode == "blackened":
            with open(f"./name_activation_dict_blackened/{name_without_extension}_{LAYER_OF_INTEREST}.pkl", "rb") as f:
                big_pickle = pickle.load(f)
        elif mode == "segmented":
            with open(f"./name_activation_dict_segmented/{name_without_extension}_{LAYER_OF_INTEREST}.pkl", "rb") as f:
                big_pickle = pickle.load(f)




        image = Image.open(image_path).convert('RGB') 
        image_tensor = test_transforms(image).unsqueeze(0).to(device)  
        with torch.no_grad():
            outputs = model(image_tensor)
            _, pred = torch.max(outputs, 1)
        class_names = os.listdir('./TestImages')  
        taxon = class_names[pred.item()]

        if taxon == "bee":
            concept = "Fur"
        elif taxon == "wasp":
            concept = "PinchWaist"
            
        print(f"Taxon is: {taxon}")    
        print(f"Concept is: {concept}")




        all_image_and_dot_values_dict = {}
        for image_name, activation in big_pickle.items():
            dot_value = compute_dot_CAV_score(concept, LAYER_OF_INTEREST, activation)
            all_image_and_dot_values_dict[image_name] = dot_value

        print(f"Dot values from least to most for mode *** {mode} ***: ")
        print(sorted(all_image_and_dot_values_dict, key=all_image_and_dot_values_dict.get)[0:5])
        print(sorted(all_image_and_dot_values_dict, key=all_image_and_dot_values_dict.get)[-5:])

        true_paths = []
        if mode == "blackened":
            for image_name in sorted(all_image_and_dot_values_dict, key=all_image_and_dot_values_dict.get)[-no_of_top_regions:]:
                true_paths.append(f"./super_blackened_images/{name_without_extension}/{image_name}")
        elif mode == "segmented":
            for image_name in sorted(all_image_and_dot_values_dict, key=all_image_and_dot_values_dict.get)[-no_of_top_regions:]:
                true_paths.append(f"./super_segmented_images/{name_without_extension}/{image_name}")

        print("**********************************************")
        image_paths = true_paths
        images = [Image.open(img).convert("RGBA") for img in image_paths]

        width, height = images[0].size
        images = [img.resize((width, height)) for img in images]
        image_arrays = [np.array(img, dtype=np.float32) for img in images]

        # Combine images while preserving transparency
        combined_array = np.zeros_like(image_arrays[0], dtype=np.float32)
        alpha_sum = np.zeros((height, width), dtype=np.float32)

        for img_array in image_arrays:
            alpha = img_array[:, :, 3] / 255.0  # Normalize alpha to [0, 1]
            combined_array[:, :, :3] += img_array[:, :, :3] * alpha[..., None]
            alpha_sum += alpha

        # Normalize combined RGB by alpha_sum and restore alpha
        combined_array[:, :, :3] = np.divide(
            combined_array[:, :, :3],
            alpha_sum[..., None],
            out=np.zeros_like(combined_array[:, :, :3]),
            where=alpha_sum[..., None] > 0
        )
        combined_array[:, :, 3] = np.clip(alpha_sum * 255, 0, 255)  # Restore alpha channel

        # Convert back to uint8 and create final image
        final_combined_array = combined_array.astype(np.uint8)
        result = Image.fromarray(final_combined_array, mode="RGBA")

        if mode == "blackened":
            os.makedirs(f"{final_result_save_dir}/blackened", exist_ok=True)
            result.save(f"{final_result_save_dir}/blackened/{name_without_extension}.png")
        elif mode == "segmented":
            os.makedirs(f"{final_result_save_dir}/segmented", exist_ok=True)
            result.save(f"{final_result_save_dir}/segmented/{name_without_extension}.png")

        result.show()
