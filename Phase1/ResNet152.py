import os
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.transforms.functional as F
import warnings
from collections import OrderedDict



warnings.filterwarnings("ignore")


image_size = "large"
num_epochs = 15


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        padding = [(max_wh - w) // 2, (max_wh - h) // 2]
        padding.extend([max_wh - w - padding[0], max_wh - h - padding[1]])
        image = F.pad(image, padding)
        return F.resize(image, (224, 224))

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  
    plt.savefig(save_path)
    plt.close()












print(model)