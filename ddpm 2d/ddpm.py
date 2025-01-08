#############################################################
# DDPM FORWARD AND SAMPLING MODULES PROCESSES ###############
# Basado en https://huggingface.co/blog/annotated-diffusion #
# Álvaro Zorrilla Carriquí ##################################
#############################################################

"""
LIBRERÍAS NECESARIAS
"""

import torch
from torch import nn, optim
from unet_ddpm2d import Unet
from tqdm import tqdm
import matplotlib.pyplot as plt


"""
Definimos la clase Diffusion_Scheme que contiene distintas funciones
para establecer el proceso de entrenamiento y el de sampleado.
"""

class Diffusion_Scheme:
    def __init__(self, image_size, device, timesteps = 1000, beta1 = 1e-4, betaT = 0.02, beta_scheme = "cosine"):
        self.timesteps = timesteps
        self.beta1 = beta1
        self.betaT = betaT
        self.image_size = image_size
        self.device = device

        if beta_scheme == "cosine":
            self.beta = self.cosine_beta_schedule().to(device)
        else:
            self.beta = self.linear_beta_schedule().to(device)

        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim = 0)

    # Dos posibles esquemas de de betas
    # Proponemos distintos conjuntos de valores de betas para el DDPM.
    # El original es un aumento lineal desde beta1=0.0001 a betaT=0.02
    # Los mejores resultados se obtienen con un esquema de coseno (https://arxiv.org/abs/2102.09672)

    def cosine_beta_schedule(self, s=0.008):
        x = torch.linspace(0, self.timesteps, self.timesteps + 1)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (s + 1) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)

    # Definimos una función que coge un tensor de imágenes x y un tensor de instantes de tiempo t
    # y devuelve las imágenes con la proporción de ruido correspondiente añadido y dicho ruido.

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low = 1, high = self.timesteps, size = (n,))

    # Función para el proceso de sampleado de imágenes con el modelo entrenado.
    def sampling(self, model, num_channels, n):
        print(f"Sampleando {n} nuevas imágenes a partir del modelo entrenado...")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, num_channels, self.image_size, self.image_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.timesteps)), position = 0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.int)
        return x

# Función que ejecuta el entrenamiento de la red U-Net del modelo 2D DDPM.
def train(dataloader, num_channels, epochs = 50, image_size = 64, device = "mps", lr = 3e-4):
    model = Unet(channels=num_channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr)
    mse = nn.MSELoss()
    diffusion = Diffusion_Scheme(image_size = image_size, device = device)

    for epoch in range(epochs):
        prog_bar = tqdm(dataloader)
        model.train()
        for count, (images, _) in enumerate(prog_bar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prog_bar.set_postfix(MSE = loss.item())
        
        if (epoch + 1) % 5 == 0:
            sampled_image = diffusion.sampling(model, num_channels=num_channels, n = 1)
            plt.imshow(sampled_image[0].permute(1, 2, 0).to('cpu').numpy())
            plt.axis('off')
            plt.show()
    
    torch.save(model.state_dict(), "model.pt")



                




