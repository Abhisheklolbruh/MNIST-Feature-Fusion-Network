import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from model import FeatureFusionNet

import os

class AddGaussianNoise:

    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, tensor):

        noise = torch.randn_like(tensor) * self.std

        return torch.clamp(
            tensor + noise,
            0,
            1
        )

os.makedirs("models", exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Training augmentation
train_transform = transforms.Compose([

    transforms.RandomRotation(12),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    transforms.ToTensor(),

    AddGaussianNoise(std=0.15),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

# Validation preprocessing
test_transform = transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.1307,),
        (0.3081,)
    )
])

train_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=train_transform
)

val_data = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=test_transform
)

# Reproducible split
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    60000,
    generator=generator
).tolist()

train_indices = indices[:50000]
val_indices = indices[50000:]

train_subset = Subset(train_data, train_indices)
val_subset = Subset(val_data, val_indices)

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=64
)

model = FeatureFusionNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    # Validation
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs} | "
        f"Loss: {total_loss/len(train_loader):.4f} | "
        f"Validation Accuracy: {accuracy:.2f}%"
    )

print("Current directory:", os.getcwd())
print("Model folder:", os.path.abspath("model"))

os.makedirs("model", exist_ok=True)

save_path = os.path.abspath(
    "model/feature_fusion_gpu_test.pth"
)

torch.save(
    model.state_dict(),
    "model/feature_fusion_noise_aug.pth"
)

print("Checkpoint saved:", os.path.isfile(save_path))
print("File size:", os.path.getsize(save_path))