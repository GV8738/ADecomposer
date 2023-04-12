import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils


import configuration
from utils import utils
from models import Vae
from dataset import prepare_dataset

def main(args):
    device = torch.device('cuda')

    # Getting dataset/dataloader
    img_transforms, mask_transforms = prepare_dataset.get_transforms(args.size)
    te_dataset = prepare_dataset.MVTecDataset(root=args.dataroot,
        target=args.category,
        train=False,
        transforms=img_transforms,
        mask_transforms=mask_transforms)
    te_dataloader = DataLoader(te_dataset, batch_size=4, shuffle=False)

    # Create model
    model = Vae.VanillaVAE(in_channels=3, latent_dim=128, hidden_dims=[32, 64, 128, 256, 512]).to(device)
    weights = torch.load(args.root + '/weights/vae_' + args.category + '_200_cycle.pth')
    model.load_state_dict(weights['model'])

    i = 0
    for (x,_,label) in te_dataloader:
        if label.sum() != 0:
            i += 1
            if i > 0:
                x = x.to(device)
                print(label)
                samples = model.generate(x)
                break
    vutils.save_image(x.cpu().data, args.root + '/samples/image.png', normalize=True)
    vutils.save_image(samples.cpu().data, args.root + '/samples/reconstruct.png', normalize=True)


if __name__ == '__main__':
    args = configuration.argparser()
    main(args)