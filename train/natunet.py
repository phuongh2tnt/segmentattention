import torch
import torch.nn as nn
import torch.nn.functional as F



class NeighborhoodAttentionBlock(nn.Module):
    def __init__(self, dim, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.dim = dim
        
        # Linear projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Relative positional biases
        self.relative_bias = nn.Parameter(torch.zeros(num_neighbors, num_neighbors))
        
        # Output linear projection
        self.fc_out = nn.Linear(dim, dim)
   def forward(self, x, neighbors):
    B, N, C = x.shape
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)
    
    attention_scores = torch.zeros(B, N, self.num_neighbors).to(x.device)
    V_neighbors = torch.zeros(B, N, self.num_neighbors, C).to(x.device)
    
    for i in range(N):
        neighbors_indices = neighbors[i]
        if len(neighbors_indices) > 0:
            Q_i = Q[:, i:i+1, :]
            K_neighbors = K[:, neighbors_indices, :]
            V_neighbors_i = V[:, neighbors_indices, :]
            
            print("Q_i shape:", Q_i.shape)
            print("K_neighbors shape:", K_neighbors.shape)
            print("V_neighbors_i shape:", V_neighbors_i.shape)
            
            attention_scores[:, i, :len(neighbors_indices)] = torch.bmm(Q_i, K_neighbors.transpose(1, 2)).squeeze(1)
            V_neighbors[:, i, :len(neighbors_indices), :] = V_neighbors_i
    
    print("Attention Scores shape:", attention_scores.shape)
    print("Relative Bias shape:", self.relative_bias.shape)
    
    attention_scores = attention_scores + self.relative_bias[:attention_scores.size(2), :attention_scores.size(2)]
    
    attention_probs = F.softmax(attention_scores / (self.dim ** 0.5), dim=-1)
    attended_values = torch.bmm(attention_probs, V_neighbors.view(B, N, -1)).view(B, N, C)
    
    print("Attended Values shape:", attended_values.shape)
    
    attended_values = self.fc_out(attended_values)
    
    return attended_values

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # BHWC
        x = x.view(B, -1, C)  # Reshape to (B, N, C) where N = H * W
        return x

class Encoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3, num_neighbors=8):
        super().__init__()
        H, W = partioned_ip_res
        self.num_neighbors = num_neighbors
        self.enc_na_blocks = nn.ModuleList([
            NeighborhoodAttentionBlock(C, num_neighbors),
            NeighborhoodAttentionBlock(2 * C, num_neighbors),
            NeighborhoodAttentionBlock(4 * C, num_neighbors)
        ])
        self.enc_patch_merge_blocks = nn.ModuleList([
            PatchMerging(C),
            PatchMerging(2 * C),
            PatchMerging(4 * C)
        ])

    def forward(self, x):
        skip_conn_ftrs = []
        neighbors = [torch.arange(x.size(1)).to(x.device) for _ in range(x.size(0))]
        for na_block, patch_merger in zip(self.enc_na_blocks, self.enc_patch_merge_blocks):
            x = na_block(x, neighbors)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

class PatchMerging(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.reduction = nn.Linear(4 * in_channels, self.out_channels, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        return x

class PatchExpansion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.expand = nn.Linear(in_channels, 2 * self.out_channels, bias=False)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int((N * 2)**0.5)
        x = self.expand(x)
        x = x.view(B, H, W, C // 2).permute(0, 2, 1, 3).contiguous()
        return x.view(B, -1, C // 2)

class Decoder(nn.Module):
    def __init__(self, C, partioned_ip_res, num_blocks=3, num_neighbors=8):
        super().__init__()
        H, W = partioned_ip_res
        self.num_neighbors = num_neighbors
        self.dec_na_blocks = nn.ModuleList([
            NeighborhoodAttentionBlock(4 * C, num_neighbors),
            NeighborhoodAttentionBlock(2 * C, num_neighbors),
            NeighborhoodAttentionBlock(C, num_neighbors)
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
        neighbors = [torch.arange(x.size(1)).to(x.device) for _ in range(x.size(0))]
        for patch_expand, na_block, enc_ftr, linear_concatter in zip(self.dec_patch_expand_blocks, self.dec_na_blocks, encoder_features, self.skip_conn_concat):
            x = patch_expand(x)
            x = torch.cat([x, enc_ftr], dim=-1)
            x = linear_concatter(x)
            x = na_block(x, neighbors)
        return x

class FinalPatchExpansion(nn.Module):
    def __init__(self, in_channels, output_res):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.expand = nn.Linear(in_channels, self.out_channels, bias=False)
        self.output_res = output_res

    def forward(self, x):
        B, N, C = x.shape
        H, W = self.output_res
        x = self.expand(x)
        x = x.view(B, H, W, C // 2).permute(0, 3, 1, 2).contiguous()
        return x

class NatUNet(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.bottleneck = NeighborhoodAttentionBlock(C * (2 ** num_blocks), num_neighbors=8)
        self.decoder = Decoder(C, (H // patch_size, W // patch_size), num_blocks)
        self.final_expansion = FinalPatchExpansion(C, (H, W))  # Pass the original image size for final expansion
        self.head = nn.Conv2d(C // 2, num_class, 1)  # Adjusted C // 2 for final expansion output

    def forward(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(x)
        return nn.functional.interpolate(x, size=(480, 480), mode='bilinear', align_corners=False)  # Upsample to original size


