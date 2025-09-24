from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])



image_size = "large"

data_dir = f'./TrainImages_{image_size}/'


dataset = datasets.ImageFolder(root=data_dir, transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)


def compute_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    for images, _ in loader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, 3, -1)  
        mean += images.mean(dim=2).sum(dim=0)
        std += images.std(dim=2).sum(dim=0)
        total_samples += batch_samples

    mean /= total_samples
    std /= total_samples
    return mean, std

mean, std = compute_mean_std(loader)
print(f"Dataset Mean: {mean.tolist()}")
print(f"Dataset Std: {std.tolist()}")

# Dataset Mean: [0.5117759704589844, 0.49503958225250244, 0.3596796989440918]
# Dataset Std: [0.20642323791980743, 0.19320522248744965, 0.1908806711435318]

