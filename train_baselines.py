
import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from baselines import MLP, CNN, count_parameters

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

os.makedirs("model", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

dataset = datasets.MNIST(
    "./data",
    train=True,
    download=True,
    transform=transform
)

generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    60000,
    generator=generator
).tolist()

train_indices = indices[:50000]
val_indices = indices[50000:]

train_subset = Subset(dataset, train_indices)
val_subset = Subset(dataset, val_indices)

train_loader = DataLoader(
    train_subset, batch_size=64, shuffle=True
)

val_loader = DataLoader(
    val_subset, batch_size=64
)


def train_model(model, name):

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    print(f"\nTraining {name}")
    print("Parameters:", count_parameters(model))

    for epoch in range(10):

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
            f"Epoch {epoch+1}/10 | "
            f"Loss: {total_loss/len(train_loader):.4f} | "
            f"Validation Accuracy: {accuracy:.2f}%"
        )

    save_path = f"model/{name.lower()}_baseline.pth"

    torch.save(model.state_dict(), save_path)

    print("Saved:", os.path.abspath(save_path))

    return model


# Train both models from fresh weights
mlp = train_model(MLP(), "MLP")
cnn = train_model(CNN(), "CNN")

print("\nBaseline training complete!")
