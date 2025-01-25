##################################
# VAE IMPLEMENTATION IN PYTORCH  #
#  BY ALVARO ZORRILLA CARRIQUI   #
##################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Implementation of VAE using convolutional and linear perceptron layers as the neural network architecture
# for the Encoder and the Decoder. This architecture can be changed following the user's preferences.  

class Encoder(nn.Module):
    def __init__(self, image_shape, latent_dim=20):
        super(Encoder, self).__init__()
        self.image_shape = image_shape
        # La primera parte del Encoder va a estar confromada por tres capas convolucionales que extraigan la
        # infromación espacial de las imágenes.
        self.encoder = nn.Sequential(  # Input: B, C, H, W
            nn.Conv2d(self.image_shape[1], 32, kernel_size=4, stride=2, padding=1),   # Output: B, 32, H//2, W//2
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),    # Output: B, 64, H//4, W//4
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),    # Output: B, 128, H//8, W//8
            nn.ReLU()
        )

        # La segunda parte del Encoder, que calcula los vectores de media y desviación típica, va a estar formada 
        # por un perceptrón que aplane el tensor para la representación del espacio latente.
        self.encoder_mean = nn.Linear(in_features=128*(self.image_shape[2]*self.image_shape[3] // 64), out_features=latent_dim) # [batch, latent_dim]
        self.encoder_log_var = nn.Linear(in_features=128*(self.image_shape[2]*self.image_shape[3] // 64), out_features=latent_dim) # [batch_latent_dim]

    def forward(self, x): # x: B, C=3, H, W
        h = self.encoder(x) # h: B, 128, H//8, W//8   Aplicamos las capas convolucionales
        h = h.view(x.size(0), -1) # h: [batch, 128*H*W//64]   Aplicamos las capas lineales para aplanar y llevar al espacio latente
        mean = self.encoder_mean(h) # mean: B, latent_dim
        log_var = self.encoder_log_var(h) # log_var: B, latent_dim
        return mean, log_var
    


class Decoder(nn.Module):
    def __init__(self, image_shape, latent_dim = 20):
        super(Decoder, self).__init__()
        self.image_shape = image_shape

        # Usamos una red lineal para pasar del espacio latente a las características de salida de las redes convolucionales
        self.fc = nn.Linear(latent_dim, 128*(self.image_shape[2]*self.image_shape[3] // 64))

        # Redes convolucionales transpuestas
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.image_shape[1], kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()  # Sigmoid activation for pixel values in [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 128, self.image_shape[2]//8, self.image_shape[3]//8)
        return self.decoder(h)


# Putting together both Encoder and Decoder classes in a single class: VariationalAutoEncoder.
class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_shape, latent_dim = 20):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(image_shape, latent_dim)
        self.decoder = Decoder(image_shape, latent_dim)

    def forward(self, x):
        mean, log_var = self.encoder(x)
        return self.decoder(mean)


# Define KL and MSE loss for the train proccess.    
def kl_loss_function(mean, log_var):
        return 0.5*torch.sum(-1 - log_var + mean**2 + torch.exp(log_var))

def mse_loss_function(input, output):
    return F.mse_loss(input, output, reduction='sum')

# Define a sampling function for training process.
def sample(mean, log_var):
    sd = torch.exp(0.5*log_var)
    return mean + sd*torch.randn_like(mean)


# Define a training function which receives as parameters the DataLoader object, the numbers of epochs (int),
# the latent space dimension (int), the image's shape (list [batch, num_channels, height, length]) and the device (mps or cuda).
# This function implements the training proccess of the neural network linked to the VAE model.
# During each epoch, the evolution of KL, MSE loss is shown. 
def train(dataloader, epochs, latent_dim, image_shape, device=torch.device('mps')):
    model = VariationalAutoEncoder(latent_dim=latent_dim, image_shape=image_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for epoch in range(epochs):
        mse_loss_sum = 0
        kl_loss_sum = 0
        loss_sum = 0
        
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        pbar.set_description(f"Epoch {epoch} - Training")
        for i, batch in pbar:
            x, _ = batch
            x = x.to(device)
            mean, log_var = model.encoder(x)
            z_sample = sample(mean, log_var)
            x_bar = model.decoder(z_sample)

            mse_loss = mse_loss_function(x, x_bar)
            kl_loss = kl_loss_function(mean, log_var)
            loss = mse_loss + kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mse_loss_sum += mse_loss.item()
            kl_loss_sum += kl_loss.item()
            loss_sum += loss.item()
            pbar.set_postfix({'mse_loss' : mse_loss_sum / (i+1), 'kl_loss' : kl_loss_sum / (i+1), 'loss' : loss_sum /(i+1)})

    return model