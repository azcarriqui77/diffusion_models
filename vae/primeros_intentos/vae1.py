##################################
# VAE IMPLEMENTATION IN PYTORCH  #
#  BY ALVARO ZORRILLA CARRIQUI   #
##################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Implementation of VAE using a multilayer perceptron as the neural network architecture
# for the Encoder and the Decoder. This architecture can be changed following the user's preferences.  

class Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=image_shape[2]*image_shape[3], out_features=100), # [batch, 100]
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=50), # [batch, 50]
            nn.ReLU()
        )

        self.encoder_mean = nn.Linear(in_features=50, out_features=latent_dim) # [batch, latent_dim]
        self.encoder_log_var = nn.Linear(in_features=50, out_features=latent_dim) # [batch_latent_dim]

    def forward(self, x):
        x = x.view(x.size(0), -1) # x: [batch, channels*length*height]
        z = self.encoder(x) # z: [batch, 50]
        mean = self.encoder_mean(z) # mean: [batch, latent_dim]
        log_var = self.encoder_log_var(z) # log_var: [batch, latent_dim]
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Decoder, self).__init__()
        self.image_shape = image_shape
        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=50), # [batch, 50]
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=100), # [batch, 100]
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=self.image_shape[1]*self.image_shape[2]*self.image_shape[3]) # [batch, channels*length*height]
        )

    def forward(self, z):
        x_bar = self.decoder(z)
        return x_bar.view(x_bar.size(0), self.image_shape[1], self.image_shape[2], self.image_shape[3]) # Restore the original image shape.


# Putting together both Encoder and Decoder classes in a single class: VariationalAutoEncoder.
class VariationalAutoEncoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = Encoder(latent_dim, image_shape)
        self.decoder = Decoder(latent_dim, image_shape)

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
