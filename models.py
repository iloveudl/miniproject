from torch import nn


class CNN(nn.Module):
    def __init__(self, conv_dropout: float = 0.25, fc_dropout: float = 0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 4),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(conv_dropout),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128),
            nn.ReLU(True),
            nn.Dropout(fc_dropout),
            nn.Linear(128, 10),
        )
        self.fc_feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, 128),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

    def forward_features(self, x):
        return self.fc_feat(self.conv(x))

