import torch
import torch.nn as nn

class FeatureFusionNet(nn.Module):

    def __init__(self):
        super().__init__()

        # CNN branch
        self.cnn = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU()
        )

        # MLP branch
        self.mlp = nn.Sequential(

            nn.Flatten(),

            nn.Linear(784, 128),
            nn.ReLU()
        )

        # Fusion layer
        self.classifier = nn.Sequential(

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(128, 10)
        )

    def forward(self, x):

        cnn_features = self.cnn(x)
        mlp_features = self.mlp(x)

        combined = torch.cat(
            [cnn_features, mlp_features],
            dim=1
        )

        return self.classifier(combined)


