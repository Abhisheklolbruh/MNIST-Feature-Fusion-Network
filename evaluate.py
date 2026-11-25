
import torch
import numpy as np

import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from model import FeatureFusionNet

# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Evaluation device:", device)

# Load model
model = FeatureFusionNet().to(device)

model.load_state_dict(
    torch.load(
        "model/feature_fusion_gpu_test.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Test preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Official test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

actual = []
predicted = []

# Evaluation
with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        actual.extend(labels.numpy())
        predicted.extend(predictions.cpu().numpy())

# Metrics
print("\nClassification Report:")

print(
    classification_report(
        actual,
        predicted,
        digits=4
    )
)

accuracy = np.mean(
    np.array(actual) == np.array(predicted)
) * 100

print(f"\nTest Accuracy: {accuracy:.2f}%")

# Confusion matrix
cm = confusion_matrix(actual, predicted)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

display.plot(cmap="Blues")

plt.title("Feature Fusion Network - MNIST")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()
