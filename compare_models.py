
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader

from baselines import MLP, CNN
from model import FeatureFusionNet

# =====================================
# 1. DEVICE
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

os.makedirs("results", exist_ok=True)

# =====================================
# 2. MODEL CONFIGURATION
# =====================================

models = {
    "MLP": (MLP, "model/mlp_baseline.pth"),

    "CNN": (CNN, "model/cnn_baseline.pth"),

    "Feature Fusion": (
        FeatureFusionNet,
        "model/feature_fusion_gpu_test.pth"
    ),

    "Noise Augmented": (
        FeatureFusionNet,
        "model/feature_fusion_noise_aug.pth"
    )
}

loaded_models = {}

for name, (architecture, checkpoint) in models.items():

    model = architecture().to(device)

    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    loaded_models[name] = model

    print(f"Loaded: {name}")

# =====================================
# 3. DATASET
# =====================================

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

# =====================================
# 4. CORRUPTION FUNCTIONS
# =====================================

def add_noise(images, sigma, generator):

    noise = torch.randn(
        images.shape,
        generator=generator
    ) * sigma

    return torch.clamp(images + noise, 0, 1)


def rotate_images(images, angle):

    return torch.stack([
        TF.rotate(image, angle, fill=0)
        for image in images
    ])


def blur_images(images):

    return torch.stack([
        TF.gaussian_blur(
            image,
            kernel_size=[5, 5],
            sigma=[1.2, 1.2]
        )
        for image in images
    ])


# =====================================
# 5. TEST CONDITIONS
# =====================================

conditions = {
    "Original": lambda x, g: x,

    "Noise 0.1": lambda x, g: add_noise(x, 0.1, g),

    "Noise 0.2": lambda x, g: add_noise(x, 0.2, g),

    "Noise 0.3": lambda x, g: add_noise(x, 0.3, g),

    "Rotation 15": lambda x, g: rotate_images(x, 15),

    "Gaussian Blur": lambda x, g: blur_images(x)
}

# =====================================
# 6. EVALUATION
# =====================================

results = {
    name: {}
    for name in loaded_models
}

for condition_name, corruption in conditions.items():

    print(f"\nTesting condition: {condition_name}")

    for model_name, model in loaded_models.items():

        correct = 0
        total = 0

        # Reset generator for fair, repeatable noise
        generator = torch.Generator().manual_seed(123)

        with torch.no_grad():

            for images, labels in test_loader:

                images = corruption(images, generator)

                images = (
                    images - 0.1307
                ) / 0.3081

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = outputs.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        accuracy = 100 * correct / total

        results[model_name][condition_name] = accuracy

        print(
            f"{model_name:20s}: {accuracy:.2f}%"
        )

# =====================================
# 7. SAVE RESULTS
# =====================================

with open(
    "results/model_comparison.txt",
    "w"
) as file:

    for model_name, scores in results.items():

        file.write(f"\n{model_name}\n")

        for condition, accuracy in scores.items():

            file.write(
                f"{condition}: {accuracy:.2f}%\n"
            )

print("\nAll evaluations completed!")

# =====================================
# 8. COMPARISON GRAPH
# =====================================

conditions_list = list(conditions.keys())

x = np.arange(len(conditions_list))

width = 0.19

plt.figure(figsize=(13, 7))

for i, (model_name, scores) in enumerate(results.items()):

    values = [
        scores[condition]
        for condition in conditions_list
    ]

    plt.bar(
        x + i * width,
        values,
        width,
        label=model_name
    )

plt.xlabel("Test Conditions")
plt.ylabel("Accuracy (%)")

plt.title("Architecture and Robustness Comparison")

plt.xticks(
    x + width * 1.5,
    conditions_list,
    rotation=20
)

plt.ylim(0, 100)

plt.legend()

plt.tight_layout()

plt.savefig(
    "results/four_model_comparison.png",
    dpi=300
)

plt.show()

print("Comparison graph saved!")
