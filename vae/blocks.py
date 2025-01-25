import torch
import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, num_heads, norm_channels, down_sample=True, 
                 attn=False):
        """
        in_channels y out_channels: dimensiones de entrada y salida del bloque
        num_heads: número de cabezas en la atención multi-cabeza
        num_layers: número de capas internas dentro del bloque, incluidas las capas convolucionales, normalización y atención
        attn: indica si se incluye un módulo de atención en el bloque
        down_sample: determina si el bloque incluye un paso de downsampling (reducción de resolución espacial)
        norm_channels: número de grupos usados en GroupNorm, una técnica de normalización
        """

        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        self.attn = attn

        self.resnet_conv1 = nn.ModuleList(
            [
                nn.Sequetial(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for i in range(num_layers)
            ]
        )

        self.resnet_conv2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
                for _ in range(num_layers)
            ]
        )

        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )

        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )

        self.down_sample_conv = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if self.down_sample else nn.Identity()

    def forward(self, x, t_emb=None):
        for i in range(self.num_layers):
            resnet_input = x
            x = self.resnet_conv1[i](x)
            x = self.resnet_conv2[i](x)
            x = x + self.residual_input_conv[i](resnet_input)

            if self.attn:
                batch_size, channels, h, w = x.shape
                in_attn = x.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                x_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                x_attn = x_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                x = x + x_attn

        x = self.down_sample_conv(x)
        return x
    

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.resnet_conv1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                            padding=1),
                )
                for i in range(num_layers + 1) ## +1 por el el primer bloque ResNet que se ejecuta antes de comenzar el ciclo principal
            ]
        )
        
        self.resnet_conv2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers + 1)
            ]
        )
        
        self.attention_norms = nn.ModuleList(
            [nn.GroupNorm(norm_channels, out_channels)
            for _ in range(num_layers)]
        )
        
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            for _ in range(num_layers)]
        )
    
    def forward(self, x):
        # Bloque independiente de ResNet antes de comenzar el ciclo principal        
        resnet_input = x
        x = self.resnet_conv1[0](x)
        x = self.resnet_conv2[0](x)
        x = x + self.residual_input_conv[0](resnet_input)
        
        # Ciclo principal
        for i in range(self.num_layers):
            # Attention Block
            batch_size, channels, h, w = x.shape
            in_attn = x.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            x_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            x_attn = x_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            x = x + x_attn
                
            # Resnet Block
            resnet_input = x
            x = self.resnet_conv1[i + 1](x)
            x = self.resnet_conv2[i + 1](x)
            x = x + self.residual_input_conv[i + 1](resnet_input)
        
        return x
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample, num_heads, num_layers, attn, norm_channels):
        super().__init__()
        self.num_layers = num_layers
        self.up_sample = up_sample
        self.attn = attn
        self.resnet_conv1 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, in_channels if i == 0 else out_channels),
                    nn.SiLU(),
                    nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, stride=1,
                              padding=1),
                )
                for i in range(num_layers)
            ]
        )
        
        self.resnet_conv2 = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(norm_channels, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                )
                for _ in range(num_layers)
            ]
        )
        if self.attn:
            self.attention_norms = nn.ModuleList(
                [
                    nn.GroupNorm(norm_channels, out_channels)
                    for _ in range(num_layers)
                ]
            )
            
            self.attentions = nn.ModuleList(
                [
                    nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
                    for _ in range(num_layers)
                ]
            )
            
        self.residual_input_conv = nn.ModuleList(
            [
                nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=1)
                for i in range(num_layers)
            ]
        )
        self.up_sample_conv = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) \
            if self.up_sample else nn.Identity()
    
    def forward(self, x):
        # Upsample
        x = self.up_sample_conv(x)
        for i in range(self.num_layers):
            # Resnet Block
            resnet_input = x
            x = self.resnet_conv1[i](x)
            x = self.resnet_conv2[i](x)
            x = x + self.residual_input_conv[i](resnet_input)
            
            # Self Attention
            if self.attn:
                batch_size, channels, h, w = x.shape
                in_attn = x.reshape(batch_size, channels, h * w)
                in_attn = self.attention_norms[i](in_attn)
                in_attn = in_attn.transpose(1, 2)
                x_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
                x_attn = x_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
                x = x + x_attn
        return x