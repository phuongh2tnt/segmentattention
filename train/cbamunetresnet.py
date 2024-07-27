import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, encoder, n_classes):
        super(UNet, self).__init__()
        
        self.encoder = encoder
        
        self.conv1 = self.encoder.conv1
        self.bn1 = self.encoder.bn1
        self.relu = self.encoder.relu
        self.maxpool = self.encoder.maxpool
        
        self.encoder1 = self.encoder.layer1
        self.encoder2 = self.encoder.layer2
        self.encoder3 = self.encoder.layer3
        self.encoder4 = self.encoder.layer4
        
        self.center = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        
        self.up4 = self.up_conv(1024, 512)
        self.up3 = self.up_conv(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.up1 = self.up_conv(128, 64)
        
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def up_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        conv1 = self.relu(self.bn1(self.conv1(x)))
        pool1 = self.maxpool(conv1)
        
        conv2 = self.encoder1(pool1)
        pool2 = F.max_pool2d(conv2, 2)
        
        conv3 = self.encoder2(pool2)
        pool3 = F.max_pool2d(conv3, 2)
        
        conv4 = self.encoder3(pool3)
        pool4 = F.max_pool2d(conv4, 2)
        
        conv5 = self.encoder4(pool4)
        
        center = self.center(conv5)
        
        up4 = self.up4(torch.cat([F.interpolate(center, scale_factor=2, mode='bilinear', align_corners=True), conv4], dim=1))
        up3 = self.up3(torch.cat([F.interpolate(up4, scale_factor=2, mode='bilinear', align_corners=True), conv3], dim=1))
        up2 = self.up2(torch.cat([F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=True), conv2], dim=1))
        up1 = self.up1(torch.cat([F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=True), conv1], dim=1))
        
        final = self.final(up1)
        
        return final

