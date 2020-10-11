import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResFeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.res1 = BasicBlock(1, 32, stride=2)
        self.res2 = BasicBlock(32, 64, stride=2)
        self.res3 = BasicBlock(64, 128, stride=2)
        self.res4 = BasicBlock(128, 256, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        o = self.res1(x)
        o = self.res2(o)
        o = self.res3(o)
        o = self.res4(o)
        o = self.avg_pool(o)
        return o.view(o.size(0), -1)


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(24, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),   # flatten directly without bottleneck 
        )

    def forward(self, x):
        out = self.f(x)
        return out


class MetricNet(nn.Module):
    def __init__(self, in_dim=4096):
        super(MetricNet, self).__init__()

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
    # m = FeatureNet()
    m = ResFeatureNet()
    o = m(x)
    print(o.size())


