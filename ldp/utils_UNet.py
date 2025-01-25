## Script que contiene varias funciones necesarias para la red U-Net en modelos de difusión
# Escrito por Álvaro Zorrilla Carriquí
# Referencia: https://github.com/explainingai-code/StableDiffusion-PyTorch/blob/main/models/blocks.py


import torch

def get_time_embedding(time_steps, temb_dim):
    """
    Esta función transforma un tensor de pasos de tiempo en un embedding usando la fórmula sinusoidal
    para embedding temporal.
    Parámetros de la función:
    - time_steps: tensor 1D de longitud igual al tamaño de batch (B)
    - temb_dim: dimensión del embedding (D)
    La función devuelve un tensor de dimension BxD que representa los B pasos de tiempo.
    """
    assert temb_dim % 2 == 0, "El embedding de tiempo debe ser múltiplo entero de 2"
    
    # factor = 10000^(2i/d_model)
    factor = 10000 ** ((torch.arange(
        start=0, end=temb_dim // 2, dtype=torch.float32, device=time_steps.device) / (temb_dim // 2))
    )
    
    # pos / factor
    # timesteps B -> B, 1 -> B, temb_dim
    t_emb = time_steps[:, None].repeat(1, temb_dim // 2) / factor
    t_emb = torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb
