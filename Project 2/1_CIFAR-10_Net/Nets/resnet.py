import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, batch_norm=True):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.elu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels) if batch_norm else nn.Sequential()
            )

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        # x = self.elu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        # x = self.sigmoid(x)
        # x = self.tanh(x)
        # x = self.elu(x)

        return x


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, if_bn=True, channel=64):
        super(ResNet18, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channel) if if_bn else nn.Sequential(),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            ResBlock(channel, channel, 1, if_bn), 
            ResBlock(channel, channel, 1, if_bn)
        )
        self.layer2 = nn.Sequential(
            ResBlock(channel, channel*2, 2, if_bn),
            ResBlock(channel*2, channel*2, 1, if_bn)
        )
        self.layer3 = nn.Sequential(
            ResBlock(channel*2, channel*4, 2, if_bn),
            ResBlock(channel*4, channel*4, 1, if_bn)
        )
        self.layer4 = nn.Sequential(
            ResBlock(channel*4, channel*8, 2, if_bn),
            ResBlock(channel*8, channel*8, 1, if_bn)
        )

        self.fc = nn.Linear(channel*8, num_classes)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
