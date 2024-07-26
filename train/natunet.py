import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # BHWC
        return x

class FinalPatchExpansion(nn.Module):
    def __init__(self, dim, out_size):
        super().__init__()
        self.expand = nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)
        self.norm = nn.LayerNorm(dim // 2)
        self.out_size = out_size

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # BCHW
        x = self.expand(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # BHWC
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

class NeighborhoodAttentionBlock(nn.Module):
    def __init__(self, dim, num_neighbors):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.dim = dim
        
        # Linear projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Relative positional biases (dummy initialization, can be learned or fixed)
        self.relative_bias = nn.Parameter(torch.zeros(num_neighbors, num_neighbors))
        
        # Output linear projection
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x, neighbors):
        B, N, C = x.shape  # B: batch size, N: number of tokens, C: dimension of tokens
        
        # Linear projections
        Q = self.query(x)  # Shape: (B, N, C)
        K = self.key(x)    # Shape: (B, N, C)
        V = self.value(x)  # Shape: (B, N, C)
        
        # Prepare to compute attention scores within neighborhoods
        attention_scores = torch.zeros(B, N, N).to(x.device)
        V_neighbors = torch.zeros(B, N, self.num_neighbors, C).to(x.device)
        
        # Compute local attention scores and values
        for i in range(N):
            neighbors_indices = neighbors[i]  # Indices of neighbors for the i-th token
            
            Q_i = Q[:, i:i+1, :]  # Shape: (B, 1, C)
            K_neighbors = K[:, neighbors_indices, :]  # Shape: (B, k, C)
            V_neighbors_i = V[:, neighbors_indices, :]  # Shape: (B, k, C)
            
            # Compute attention scores
            attention_scores[:, i, neighbors_indices] = torch.bmm(Q_i, K_neighbors.transpose(1, 2)).squeeze(1) + self.relative_bias
            
            # Store values
            V_neighbors[:, i, :] = torch.cat([V_neighbors_i, torch.zeros(B, self.num_neighbors - V_neighbors_i.size(1), C).to(x.device)], dim=1)
        
        # Normalize attention scores
        attention_probs = F.softmax(attention_scores / (self.dim ** 0.5), dim=-1)
        
        # Weighted sum of values
        attended_values = torch.bmm(attention_probs, V_neighbors.view(B, N, self.num_neighbors * C))  # Shape: (B, N, C)
        
        # Apply output projection
        attended_values = self.fc_out(attended_values)
        
        return attended_values

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
        # Example neighbors for simplicity; replace with actual neighbors computation
        neighbors = [torch.arange(x.size(1)).to(x.device) for _ in range(x.size(0))]
        for na_block, patch_merger in zip(self.enc_na_blocks, self.enc_patch_merge_blocks):
            x = na_block(x, neighbors)
            skip_conn_ftrs.append(x)
            x = patch_merger(x)
        return x, skip_conn_ftrs

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

class NatUNet(nn.Module):
    def __init__(self, H, W, ch, C, num_class, num_blocks=3, patch_size=4, num_neighbors=8):
        super().__init__()
        self.patch_embed = PatchEmbedding(ch, C, patch_size)
        self.encoder = Encoder(C, (H // patch_size, W // patch_size), num_blocks, num_neighbors)
        self.bottleneck = NeighborhoodAttentionBlock(C * (2 ** num_blocks), num_neighbors)
        self.decoder = Decoder(C, (H // patch_size, W // patch_size), num_blocks, num_neighbors)
        self.final_expansion = FinalPatchExpansion(C, (H, W))  # Pass the original image size for final expansion
        self.head = nn.Conv2d(C // 2, num_class, 1)  # Adjusted C // 2 for final expansion output

    def forward(self, x):
        x = self.patch_embed(x)
        x, skip_ftrs = self.encoder(x)
        x = self.bottleneck(x, [torch.arange(x.size(1)).to(x.device) for _ in range(x.size(0))])
        x = self.decoder(x, skip_ftrs[::-1])
        x = self.final_expansion(x)
        x = self.head(x.permute(0, 3, 1, 2))
        return nn.functional.interpolate(x, size=(480, 480), mode='bilinear', align_corners=False)  # Upsample to original size
