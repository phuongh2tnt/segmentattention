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

class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3):
        super().__init__()
        H, W = partioned_ip_res
        self.dec_swin_blocks = nn.ModuleList([
            SwinBlock(4 * C, (H // 4, W // 4)),
            SwinBlock(2 * C, (H // 2, W // 2)),
            SwinBlock(C, (H, W))
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
        for patch_expand, swin_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_swin_blocks, encoder_features, self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = swin_block(x)
        return x

class SwinUNet(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = SwinBlock(C * (2 ** num_blocks), (H // (patch_size * (2 ** num_blocks)), W // (patch_size * (2 ** num_blocks))))
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

