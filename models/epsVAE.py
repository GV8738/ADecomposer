import copy
import torch
from models import base
from torch import nn
from torch.nn import functional as F
from models.types_ import *

class εVAE(base.BaseVAE):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(εVAE, self).__init__()

        self.latent_dim = latent_dim
        modules = []
        if hidden_dims is not None:
            hidden_dims = [32, 64, 128, 256, 512]

        #Build Encoder
        for h_dim in hidden_dims:
            modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,kernel_size=3,stride=2,padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 64, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 64, latent_dim)

        #Build Decoders
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*64)
        hidden_dims.reverse()
        for i in range(len(hidden_dims) -1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i],hidden_dims[i+1],kernel_size=3,stride=2,padding=1,output_padding=1),
                nn.BatchNorm2d(hidden_dims[i+1]),
                nn.LeakyReLU()))
        self.decoderA = nn.Sequential(*modules)
        self.decoderB = copy.deepcopy(self.decoderA)
        self.final_layerA = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],hidden_dims[-1],kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1],out_channels=3,kernel_size=3,padding=1),
            nn.Tanh())
        self.final_layerB = copy.deepcopy(self.final_layerA)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        '''Maps the latent codes onto the image space
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W
        '''
        result = self.decoder_input(z)
        result = result.view(-1, 512, 8, 8)
        y = self.decoderA(result)
        y = self.final_layerA(y)
        ε = self.decoderB(result)
        ε = self.final_layerB(ε)

        return y, ε

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        '''Reparameterization trick to sample from N(μ, σ) from N(0,1)
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        '''
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)

        return eps*std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        y, ε = self.decode(z)
        return [y, ε, input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> List[Tensor]:
        '''
        Computes the VAE loss function as:
        KL(N(μ,σ), N(0,1)) = log(1/σ) + (σ^2 + μ^2)/2 - 1/2
        :param args:
        :param kwargs:
        :return:
        '''
        y = args[0]
        ε = args[1]
        input = args[2]
        mu = args[3]
        log_var = args[4]

        recons = y + ε
        kld_weight = kwargs['M_N'] #Acount for the minibatch samples from the dataset
        γ = kwargs['gamma']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight*kld_loss + γ*ε.norm(p=1)

        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        '''
        Samples from the latent space and return the corresponding image space map
        :param num_samples: (int) Number of samples
        :param current_device: (int) Device to run the model
        :param kwargs: (Tensor)
        :return:  (Tensor) images
        '''
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        '''
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :returns: (Tensor) [B x C x H x W]
        '''
        y, ε = self.forward(x)
        return y, ε
