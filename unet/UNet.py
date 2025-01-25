import torch
import torch.nn as nn
from utils import get_time_embedding
from blocks import DownBlock, MidBlock, UpBlock

"""
    Parámetros:
    - image_channels: número de canales de la imagen (int)
    - down_channels: lista con los canales de las capas de downsampling (list)
    - mid_channels: lista con los canales de las capas de midblock (list)
    - time_emb_dim: dimensión del embedding de tiempo (int)
    - down_sample: lista de booleanos que indica si se hace downsampling en cada bloque (list)
    - num_down_layers: número de capas en los bloques de downsampling (int)
    - num_mid_layers: número de capas en los bloques de midblock (int)
    - num_up_layers: número de capas en los bloques de upsampling (int)
    - attn_down: lista de booleanos que indica si se usa atención en cada bloque de downsampling (list)
    - norm_channels: número de canales en la normalización (int)
    - num_heads: número de cabezas en la atención (int)
    - conv_out_channels: número de canales en la capa de salida (int)
"""

class Unet(nn.Module):
    def __init__(self, image_channels, down_channels, mid_channels, time_emb_dim,
                 down_sample, num_down_layers, num_mid_layers, num_up_layers, 
                 attn_down, norm_channels, num_heads, conv_out_channels):
        super().__init__()
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.t_emb_dim = time_emb_dim
        self.down_sample = down_sample
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        self.attns = attn_down
        self.norm_channels = norm_channels
        self.num_heads = num_heads
        self.conv_out_channels = conv_out_channels
        
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-2]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attns) == len(self.down_channels) - 1
        
        # Initial projection from sinusoidal time embedding
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(image_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.downs.append(DownBlock(self.down_channels[i], self.down_channels[i + 1], self.t_emb_dim,
                                        down_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_down_layers,
                                        attn=self.attns[i], norm_channels=self.norm_channels))
        
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i + 1], self.t_emb_dim,
                                      num_heads=self.num_heads,
                                      num_layers=self.num_mid_layers,
                                      norm_channels=self.norm_channels))
        
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels) - 1)):
            self.ups.append(UpBlock(self.down_channels[i] * 2, self.down_channels[i - 1] if i != 0 else self.conv_out_channels,
                                    self.t_emb_dim, up_sample=self.down_sample[i],
                                        num_heads=self.num_heads,
                                        num_layers=self.num_up_layers,
                                        norm_channels=self.norm_channels))
        
        self.norm_out = nn.GroupNorm(self.norm_channels, self.conv_out_channels)
        self.conv_out = nn.Conv2d(self.conv_out_channels, image_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t):
        # Shapes assuming downblocks are [C1, C2, C3, C4]
        # Shapes assuming midblocks are [C4, C4, C3]
        # Shapes assuming downsamples are [True, True, False]
        # B x C x H x W
        out = self.conv_in(x)
        # B x C1 x H x W
        
        # t_emb -> B x t_emb_dim
        t_emb = get_time_embedding(torch.as_tensor(t).long(), self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        down_outs = []
        
        for idx, down in enumerate(self.downs):
            down_outs.append(out)
            out = down(out, t_emb)
        # down_outs  [B x C1 x H x W, B x C2 x H/2 x W/2, B x C3 x H/4 x W/4]
        # out B x C4 x H/4 x W/4
        
        for mid in self.mids:
            out = mid(out, t_emb)
        # out B x C3 x H/4 x W/4
        
        for up in self.ups:
            down_out = down_outs.pop()
            out = up(out, down_out, t_emb)
            # out [B x C2 x H/4 x W/4, B x C1 x H/2 x W/2, B x 16 x H x W]
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)
        # out B x C x H x W
        return out