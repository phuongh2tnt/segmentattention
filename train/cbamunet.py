import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class CBAMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cbam = CBAM(dim)

    def forward(self, x):
        return self.cbam(x)

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

class PatchExpansion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.expand = nn.Linear(dim, 2 * dim, bias=False)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = x.view(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * 2, W * 2, C // 2)
        x = self.norm(x)
        return x

class FinalPatchExpansion(nn.Module):
    def __init__(self, dim, out_size):
        super().__init__()
        self.expand = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim // 2)
        self.out_size = out_size

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # Change to BCHW format
        x = self.expand(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # Change back to BHWC format
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.enc_cbam_blocks = nn.ModuleList([
            CBAMBlock(C),
            CBAMBlock(2 * C),
            CBAMBlock(4 * C)
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2 * C),
            PatchMerging(4 * C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        for cbam_block, patch_merger in zip(self.enc_cbam_blocks, self.enc_patch_merge_blocks):
            x = cbam_block(x)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.dec_cbam_blocks = nn.ModuleList([
            CBAMBlock(4 * C),
            CBAMBlock(2 * C),
            CBAMBlock(C)
        ])
        self.dec_patch_expand_blocks = nn.ModuleList([
            PatchExpansion(8 * C),
            PatchExpansion(4 * C),
            PatchExpansion(2 * C)
        ])
        self.skip_conn_concat = nn.ModuleList([
            nn.Linear(8 * C, 4 * C),
            nn.Linear(4 * C, 2 * C),
            nn.Linear(2 * C, C)
        ])

    def forward(self, x, encoder_features):
        for patch_expand, cbam_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_cbam_blocks, encoder_features, self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = cbam_block(x)
        return x

class CBAMUNet(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = CBAMBlock(C * (2 ** num_blocks))
        self.decoder = Decoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C, (H, W))  # Pass the original image size for final expansion
        self.head = nn.Conv2d(C // 2, num_class, 1)  # Adjusted C // 2 for final expansion output

    def forward(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(x.permute(0, 3, 1, 2))
        return nn.functional.interpolate(x, size=(480, 480), mode='bilinear', align_corners=False)  # Upsample to original size
