import torch
import torchvision.utils as vutils


import configuration
from utils import utils
from models import Vae

def main(args):
    device = torch.device('cuda')

    # Create model
    model = Vae.VanillaVAE(in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256, 512]).to(device)
    weights = torch.load(args.root + '/weights/vae100.pth')
    model.load_state_dict(weights['model'])

    samples = model.sample(num_samples=4, current_device=0)
    vutils.save_image(samples.cpu().data, args.root+'/samples/image.png', normalize=True)


if __name__ == '__main__':
    args = configuration.argparser()
    main(args)