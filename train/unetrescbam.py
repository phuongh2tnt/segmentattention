import torch
import torch.nn as nn
import torch.nn.functional as F
from rescbam import resnet50_cbam

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        return x

class UNet(nn.Module):
    def __init__(self, encoder, n_classes):
        super(UNet, self).__init__()
        self.encoder = encoder

        self.up1 = UpConv(2048, 1024)
        self.up2 = UpConv(1024, 512)
        self.up3 = UpConv(512, 256)
        self.up4 = UpConv(256, 64)
        self.out_conv = nn.Conv2d(128, n_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x)

        x = self.out_conv(x)
        return x

def resnet50_cbam_unet(pretrained=True, n_classes=2):
    encoder = resnet50_cbam(pretrained=pretrained)
    model = UNet(encoder, n_classes)
    return model
