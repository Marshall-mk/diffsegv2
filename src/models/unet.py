import torch
import torch.nn as nn
from .blocks import SinusoidalPositionEmbedding, ResidualBlock, AttentionBlock


class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, time_emb_dim: int = 256):
        super().__init__()
        self.time_embedding = SinusoidalPositionEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.enc_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.enc_blocks = nn.ModuleList([
            ResidualBlock(64, 64, time_emb_dim),
            ResidualBlock(64, 128, time_emb_dim),
            ResidualBlock(128, 128, time_emb_dim),
            ResidualBlock(128, 256, time_emb_dim),
            ResidualBlock(256, 256, time_emb_dim),
        ])
        
        # Middle
        self.mid_block1 = ResidualBlock(256, 256, time_emb_dim)
        self.mid_attn = AttentionBlock(256)
        self.mid_block2 = ResidualBlock(256, 256, time_emb_dim)
        
        # Decoder
        self.dec_blocks = nn.ModuleList([
            ResidualBlock(512, 256, time_emb_dim),
            ResidualBlock(512, 128, time_emb_dim),
            ResidualBlock(256, 128, time_emb_dim),
            ResidualBlock(192, 64, time_emb_dim),
            ResidualBlock(128, 64, time_emb_dim),
        ])
        
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
        # Downsampling and upsampling
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)
        
        h = self.enc_conv(x)
        skip_connections = []
        
        # Encoder
        for i, block in enumerate(self.enc_blocks):
            h = block(h, time_emb)
            skip_connections.append(h)
            if i in [1, 3]:  # Downsample after certain blocks
                h = self.downsample(h)
        
        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Decoder
        for i, block in enumerate(self.dec_blocks):
            if i in [0, 2]:  # Upsample before certain blocks
                h = self.upsample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, time_emb)
        
        return self.final_conv(h)