import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.conv5 = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x)))
        x3 = self.relu(self.bn3(self.conv3(x)))
        x4 = self.relu(self.bn4(self.conv4(x)))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.relu(self.bn5(self.conv5(x)))

        return x

class DeepLabV3_CBAM(nn.Module):
    def __init__(self, n_classes):
        super(DeepLabV3_CBAM, self).__init__()
        self.backbone = models.resnet50(pretrained=True)

        # Extract layers from ResNet50
        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1 = self.backbone.layer1
        self.layer2 = self.backbone.layer2
        self.layer3 = self.backbone.layer3
        self.layer4 = self.backbone.layer4

        self.aspp = ASPP(2048, 256)

        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()

        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1, bias=False),  # Fixed channels here
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.size()[2:]

        # Backbone
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        # CBAM
        x4 = self.ca(x4) * x4
        x4 = self.sa(x4) * x4

        # ASPP
        x5 = self.aspp(x4)

        # Decoder
        x5 = F.interpolate(x5, size=x1.size()[2:], mode='bilinear', align_corners=False)
        x1 = F.interpolate(x1, size=x5.size()[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x5], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x
