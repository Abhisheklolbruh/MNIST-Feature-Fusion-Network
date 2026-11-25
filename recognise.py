
import torch
import numpy as np

from model import FeatureFusionNet
from segment import segment_digits

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = FeatureFusionNet().to(device)

model.load_state_dict(
    torch.load(
        "model/feature_fusion_noise_aug.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


def recognize(image_path):

    digits, output = segment_digits(image_path)

    predictions = []

    for digit in digits:

        # Convert OpenCV image to tensor
        image = digit.astype(
            np.float32
        ) / 255.0

        # MNIST normalization
        image = (
            image - 0.1307
        ) / 0.3081

        tensor = torch.from_numpy(
            image
        ).unsqueeze(0).unsqueeze(0)

        tensor = tensor.to(device)

        with torch.no_grad():

            logits = model(tensor)

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, prediction = (
                probabilities.max(dim=1)
            )

        predictions.append({
            "digit": prediction.item(),
            "confidence": confidence.item()
        })

    number = "".join(
        str(item["digit"])
        for item in predictions
    )

    return number, predictions, output


if __name__ == "__main__":

    number, predictions, output = recognize(
        "handwriting.png"
    )

    print("\nRecognized number:", number)

    for i, item in enumerate(predictions):

        print(
            f"Digit {i+1}: {item['digit']} | "
            f"Confidence: {item['confidence']*100:.2f}%"
        )
