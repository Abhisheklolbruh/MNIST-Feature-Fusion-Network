
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

from model import FeatureFusionNet

# ---------------------------------
# 1. Device and model
# ---------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

model = FeatureFusionNet().to(device)

model.load_state_dict(
    torch.load(
        "model/feature_fusion_noise_aug.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# ---------------------------------
# 2. Load test dataset
# ---------------------------------

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# ---------------------------------
# 3. Corruption functions
# ---------------------------------

def add_noise(images, sigma):

    noise = torch.randn_like(images) * sigma

    return torch.clamp(images + noise, 0, 1)


def rotate_images(images, angle):

    rotated = [
        TF.rotate(image, angle, fill=0)
        for image in images
    ]

    return torch.stack(rotated)


def blur_images(images):

    return TF.gaussian_blur(
        images,
        kernel_size=[5, 5],
        sigma=[1.2, 1.2]
    )


# ---------------------------------
# 4. Evaluation function
# ---------------------------------

def evaluate(condition):

    correct = 0
    total = 0

    for images, labels in test_loader:

        # Apply corruption to raw [0,1] images
        images = condition(images)

        # Normalize exactly as during training
        images = (images - 0.1307) / 0.3081

        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return 100 * correct / total


# ---------------------------------
# 5. Run experiments
# ---------------------------------

results = {}

print("\nStarting robustness experiments...\n")

# Original images
results["Original"] = evaluate(
    lambda x: x
)

# Gaussian noise
for sigma in [0.1, 0.2, 0.3]:

    name = f"Noise {sigma}"

    results[name] = evaluate(
        lambda x, s=sigma: add_noise(x, s)
    )

# Rotation
results["Rotation 15"] = evaluate(
    lambda x: rotate_images(x, 15)
)

# Blur
results["Gaussian Blur"] = evaluate(
    blur_images
)

# ---------------------------------
# 6. Display results
# ---------------------------------

print("\nROBUSTNESS RESULTS")

for name, accuracy in results.items():

    print(
        f"{name:20s}: {accuracy:.2f}%"
    )

# ---------------------------------
# 7. Plot results
# ---------------------------------

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(11, 6))

plt.bar(
    results.keys(),
    results.values()
)

plt.ylabel("Accuracy (%)")
plt.xlabel("Test condition")

plt.title(
    "Feature Fusion Network - Robustness Analysis"
)

plt.xticks(rotation=25)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "results/robustness_comparison.png",
    dpi=300
)

plt.show()

# ---------------------------------
# 8. Save numerical results
# ---------------------------------

with open(
    "results/robustness_results.txt",
    "w"
) as file:

    for name, accuracy in results.items():

        file.write(
            f"{name}: {accuracy:.2f}%\n"
        )

print("\nRobustness testing complete!")
