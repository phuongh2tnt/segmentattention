import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed

class CustomPatchEmbedding(PatchEmbed):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

class FinalPatchExpansion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Định nghĩa kích thước đầu vào và đầu ra cho lớp Linear
        self.expand = nn.Linear(dim, 2 * dim)
        # Lớp chuẩn hóa
        self.norm = nn.LayerNorm(2 * dim)

    def forward(self, x):
        # Mở rộng kích thước của đầu ra
        x = self.expand(x)
        B, H, W, C = x.shape
        # Thay đổi kích thước để phù hợp với kích thước mong muốn
        x = x.view(B, H * 2, W * 2, C // 2)
        # Áp dụng chuẩn hóa
        x = self.norm(x)
        return x

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
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=16):
        super().__init__()
        self.patch_embed = CustomPatchEmbedding(img_size=(H, W), patch_size=patch_size, in_chans=ch, embed_dim=C)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = SwinBlock(C * (2 ** num_blocks), (H // (patch_size * (2 ** num_blocks)), W // (patch_size * (2 ** num_blocks))))
        self.decoder = Decoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C)
        self.head = nn.Conv2d(C, num_class, 1, padding='same')

    def forward(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(x.permute(0, 3, 1, 2))
        return x
