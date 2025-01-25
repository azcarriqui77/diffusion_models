## SCRIPT CON LA IMPLEMENTACIÓN DE LA ARQUITECTURA VAE

# Escrito por Álvaro Zorrilla Carriquí
# Basado en el código de ExplainingAI-Code: https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/models/vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import DownBlock, UpBlock, MidBlock

# Parámetros de la clase VAE
"""
    image_channels: número de canales de la imagen de entrada (int)
    down_channels: lista de dimensiones de los canales de entrada y salida de los bloques de down (list)
    mid_channels: lista de dimensiones de los canales de entrada y salida de los bloques de mid (list)
    down_sample: lista de booleanos que indica si se realiza downsample en cada bloque de down (list)
    num_down_layers: número de capas en cada bloque de down (int)
    num_mid_layers: número de capas en cada bloque de mid (int)
    num_up_layers: número de capas en cada bloque de up (int)
    latent_channels: número de canales en la representación latente (int)
    norm_channels: número de grupos usados en GroupNorm, una técnica de normalización (int)
    num_heads: número de cabezas en la atención multi-cabeza (int)
    attn_down: lista de booleanos que indica si se incluye un módulo de atención en cada bloque de down (list)
"""

class VAE(nn.Module):
    def __init__(self, image_channels, down_channels, mid_channels, down_sample, num_down_layers,
                 num_mid_layers, num_up_layers, latent_channels, norm_channels, num_heads, attn_down):
        super().__init__()
        self.down_channels = down_channels
        self.mid_channels = mid_channels
        self.down_sample = down_sample
        self.num_down_layers = num_down_layers
        self.num_mid_layers = num_mid_layers
        self.num_up_layers = num_up_layers
        self.attn_down = attn_down
        self.latent_channels = latent_channels
        self.norm_channels = norm_channels
        self.num_heads = num_heads

        self.up_sample = list(reversed(self.down_sample))

        # Validación de la información de los canales
        assert self.mid_channels[0] == self.down_channels[-1]
        assert self.mid_channels[-1] == self.down_channels[-1]
        assert len(self.down_sample) == len(self.down_channels) - 1
        assert len(self.attn_down) == len(self.down_channels) - 1

        # Encoder
        self.encoder_conv_in = nn.Conv2d(image_channels, self.down_channels[0], kernel_size=3, padding=(1,1))

        self.encoder_layers = nn.ModuleList(
            [
                DownBlock(self.down_channels[i], self.down_channels[i+1], down_sample=self.down_sample[i], 
                          num_heads=self.num_heads, num_layers=self.num_down_layers, attn=self.attn_down[i], 
                          norm_channels=self.norm_channels) for i in range(len(self.down_channels) - 1)
            ]
        )

        self.encoder_mids = nn.ModuleList(
            [
                MidBlock(self.mid_channels[i], self.mid_channels[i+1], num_layers=self.num_mid_layers, 
                         norm_channels=self.norm_channels, num_heads=self.num_heads) for i in range(len(self.mid_channels) - 1)
            ]
        )

        self.encoder_conv_out = nn.Sequential(
            nn.GroupNorm(self.norm_channels, self.down_channels[-1]),
            nn.SiLU(),
            nn.Conv2d(self.down_channels[-1], 2*self.latent_channels, kernel_size=3, padding=1)
        )  
        
        self.pre_quant_conv = nn.Conv2d(2*self.latent_channels, 2*self.latent_channels, kernel_size=1) # Predecimos mean & logvar juntos

        # Decoder

        self.post_quant_conv = nn.Conv2d(self.latent_channels, self.latent_channels, kernel_size=1)
        self.decoder_conv_in = nn.Conv2d(self.latent_channels, self.mid_channels[-1], kernel_size=3, padding=(1, 1))
        
        self.decoder_mids = nn.ModuleList([])
        for i in reversed(range(1, len(self.mid_channels))):
            self.decoder_mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i - 1], num_heads=self.num_heads,
                                              num_layers=self.num_mid_layers, norm_channels=self.norm_channels))
        
        self.decoder_layers = nn.ModuleList([])
        for i in reversed(range(1, len(self.down_channels))):
            self.decoder_layers.append(UpBlock(self.down_channels[i], self.down_channels[i - 1], up_sample=self.down_sample[i - 1],
                                               num_heads=self.num_heads, num_layers=self.num_up_layers, attn=self.attn_down[i - 1],
                                               norm_channels=self.norm_channels))
        
        self.decoder_conv_out = nn.Sequential(
            nn.GroupNorm(self.norm_channels, self.down_channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.down_channels[0], image_channels, kernel_size=3, padding=1)
        ) 

    def encoder(self, x):
        x = self.encoder_conv_in(x)
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
        for i, layer in enumerate(self.encoder_mids):
            x = layer(x)
        x = self.encoder_conv_out(x)
        x = self.pre_quant_conv(x)
        mean, log_var = torch.chunk(x, 2, dim=1)
        return mean, log_var
    
    def decoder(self, z):
        x_bar = self.post_quant_conv(z)
        x_bar = self.decoder_conv_in(x_bar)
        for i, layer in enumerate(self.decoder_mids):
            x_bar = layer(x_bar)
        for i, layer in enumerate(self.decoder_layers):
            x_bar = layer(x_bar)
        x_bar = self.decoder_conv_out(x_bar)
        return x_bar
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = mean + torch.exp(0.5*log_var)*torch.randn(mean.shape).to(device=x.device)
        x_bar = self.decoder(z)
        return x_bar
    

# Define KL and MSE loss for the train proccess.    
def kl_loss_function(mean, log_var):
        return 0.5*torch.sum(-1 - log_var + mean**2 + torch.exp(log_var))

def mse_loss_function(input, output):
    return F.mse_loss(input, output, reduction='sum')
        

        