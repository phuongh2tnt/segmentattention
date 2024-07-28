import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DeepLabV3SE(nn.Module):
    def __init__(self, num_classes=21):
        super(DeepLabV3SE, self).__init__()
        self.deeplabv3 = deeplabv3_resnet50(pretrained=True)
        self.se1 = SELayer(64, 16)
        self.se2 = SELayer(256, 16)
        self.se3 = SELayer(512, 16)
        self.se4 = SELayer(1024, 16)
        self.se5 = SELayer(2048, 16)

        # Modify the classifier to output the desired number of classes
        self.deeplabv3.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.deeplabv3.backbone.conv1(x)
        x = self.deeplabv3.backbone.bn1(x)
        x = self.deeplabv3.backbone.relu(x)
        x = self.se1(x)
        x = self.deeplabv3.backbone.maxpool(x)

        x = self.deeplabv3.backbone.layer1(x)
        x = self.se2(x)
        x = self.deeplabv3.backbone.layer2(x)
        x = self.se3(x)
        x = self.deeplabv3.backbone.layer3(x)
        x = self.se4(x)
        x = self.deeplabv3.backbone.layer4(x)
        x = self.se5(x)

        x = self.deeplabv3.classifier(x)
        x = nn.functional.interpolate(x, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        return x
