import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, input_resolution, shift_size=0):
        super().__init__()
        self.swtb1 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution)
        self.swtb2 = SwinTransformerBlock(dim=dim, input_resolution=input_resolution, shift_size=shift_size)

    def forward(self, x):
        return self.swtb2(self.swtb1(x))

class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.enc_swin_blocks = nn.ModuleList([
            SwinBlock(C, (H, W)),
            SwinBlock(2 * C, (H // 2, W // 2)),
            SwinBlock(4 * C, (H // 4, W // 4))
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2 * C),
            PatchMerging(4 * C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for swin_block, patch_merger in zip(self.enc_swin_blocks, self.enc_patch_merge_blocks):
            x = swin_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.aspp1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.aspp2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=18, dilation=18)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, stride=1)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1, stride=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        return self.relu(x)

class SwinDeepLabV3(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = SwinBlock(C * (2 ** num_blocks), (H // (patch_size * (2 ** num_blocks)), W // (patch_size * (2 ** num_blocks))))
        self.aspp = ASPP(C * (2 ** num_blocks), 256)  # Adjust ASPP output channels as needed
        self.final_conv = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        x = self.patch_embed(x)
        x, _ = self.encoder(x)
        x = self.bottleneck(x)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # Change to BCHW format
        x = self.aspp(x)
        x = self.final_conv(x)
        return nn.functional.interpolate(x, size=(H * 4, W * 4), mode='bilinear', align_corners=False)  # Upsample to original size

# Example usage:
# model = SwinDeepLabV3(H=480, W=480, ch=3, C=64, num_class=21, num_blocks=3, patch_size=4)
# input_tensor = torch.randn(1, 3, 480, 480)
# output = model(input_tensor)
