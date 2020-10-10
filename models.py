import torch
import torch.nn as nn


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.f = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),   # flatten directly without bottleneck 
        )

    def forward(self, x):
        out = self.f(x)
        return out


class MetricNet(nn.Module):
    def __init__(self, in_dim=4096):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.fc(x)


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 64)
    m = FeatureNet()
    o = m(x)
    print(o.size())
